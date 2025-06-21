from diffusers import UNet2DConditionModel, DDIMScheduler
import torch.nn.functional as F
import torch.nn as nn
import torch
import types 
import argparse 





def predict_noise(unet, noisy_latents, text_embeddings, uncond_embedding, timesteps, 
    guidance_scale=1.0, unet_added_conditions=None, uncond_unet_added_conditions=None
):
    CFG_GUIDANCE = guidance_scale > 1

    if CFG_GUIDANCE:
        model_input = torch.cat([noisy_latents] * 2) 
        embeddings = torch.cat([uncond_embedding, text_embeddings]) 
        timesteps = torch.cat([timesteps] * 2) 

        if unet_added_conditions is not None:
            assert uncond_unet_added_conditions is not None 
            condition_input = {}
            for key in unet_added_conditions.keys():
                condition_input[key] = torch.cat(
                    [uncond_unet_added_conditions[key], unet_added_conditions[key]] # should be uncond, cond, check the order  
                )
        else:
            condition_input = None 

        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=condition_input).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 
    else:
        
        model_input = noisy_latents 
        embeddings = text_embeddings
        timesteps = timesteps    
        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=unet_added_conditions).sample

    return noise_pred 


def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample






class SDGuidance(nn.Module):
    def __init__(self, args, real_unet, fake_unet, num_train_timesteps):
        super().__init__()
        self.args = args 

        # self.real_unet = UNet2DConditionModel.from_pretrained(
        #     args.pretrained_teacher_model,
        #     subfolder="unet"
        # ).float().to('cuda')
        self.real_unet = real_unet
        # self.real_unet.requires_grad_(False)

        # self.fake_unet = UNet2DConditionModel.from_pretrained(
        #     args.pretrained_teacher_model,
        #     subfolder="unet"
        # ).float().to('cuda')

        # self.fake_unet.requires_grad_(True)
        self.fake_unet = fake_unet

        # we move real unet to half precision
        # as we don't backpropagate through it
        # if args.use_fp16:
        #     self.real_unet = self.real_unet.to(torch.float32)
        #     self.fake_unet = self.fake_unet.to(torch.float32)

        self.scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="scheduler"
        )
        
        alphas_cumprod = self.scheduler.alphas_cumprod
        alphas_cumprod = alphas_cumprod.to('cuda')
        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod
        )
        
        self.num_train_timesteps = num_train_timesteps
        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)
        
        self.real_guidance_scale = args.real_guidance_scale 
        self.fake_guidance_scale = args.fake_guidance_scale

        assert self.fake_guidance_scale == 1, "no guidance for fake"

        self.use_fp16 = args.use_fp16


        self.sdxl = args.sdxl 



    def compute_distribution_matching_loss(
        self, 
        latents,
        text_embedding,
        uncond_embedding,
        unet_added_conditions,
        uncond_unet_added_conditions
    ):
        original_latents = latents 
        batch_size = latents.shape[0]
        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step, 
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size], 
                device=latents.device,
                dtype=torch.long
            )

            noise = torch.randn_like(latents)

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # run at full precision as autocast and no_grad doesn't work well together 
            pred_fake_noise = predict_noise(
                self.fake_unet, noisy_latents, text_embedding, uncond_embedding, 
                timesteps, guidance_scale=self.fake_guidance_scale,
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions
            )  

            pred_fake_image = get_x0_from_noise(
                noisy_latents.double(), pred_fake_noise.double(), self.alphas_cumprod.double(), timesteps
            )

            if self.use_fp16:
                if self.sdxl:
                    bf16_unet_added_conditions = {} 
                    bf16_uncond_unet_added_conditions = {} 

                    for k,v in unet_added_conditions.items():
                        bf16_unet_added_conditions[k] = v.to(torch.float32)
                    for k,v in uncond_unet_added_conditions.items():
                        bf16_uncond_unet_added_conditions[k] = v.to(torch.float32)
                else:
                    bf16_unet_added_conditions = unet_added_conditions 
                    bf16_uncond_unet_added_conditions = uncond_unet_added_conditions

                pred_real_noise = predict_noise(
                    self.real_unet, noisy_latents.to(torch.float32), text_embedding.to(torch.float32), 
                    uncond_embedding.to(torch.float32), 
                    timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=bf16_unet_added_conditions,
                    uncond_unet_added_conditions=bf16_uncond_unet_added_conditions
                ) 
            else:
                pred_real_noise = predict_noise(
                    self.real_unet, noisy_latents, text_embedding, uncond_embedding, 
                    timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions
                )

            pred_real_image = get_x0_from_noise(
                noisy_latents.double(), pred_real_noise.double(), self.alphas_cumprod.double(), timesteps
            )     

            p_real = (latents - pred_real_image)
            p_fake = (latents - pred_fake_image)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) 
            grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(), reduction="mean")         

        loss_dict = {
            "loss_dm": loss 
        }

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach().float(),
            "dmtrain_pred_real_image": pred_real_image.detach().float(),
            "dmtrain_pred_fake_image": pred_fake_image.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item()
        }

        return loss_dict, dm_log_dict



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_teacher_model", type=str, default="/maindata/data/shared/public/yang.zhang/models/blue_pencil-XL-v7.0.0")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--use_fp16", action="store_true")
    # parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1)
    parser.add_argument("--sdxl", action="store_true")
    args = parser.parse_args()


    return args 

if __name__ == "__main__":
    args = parse_args()
    sdguidance = SDGuidance(args)


