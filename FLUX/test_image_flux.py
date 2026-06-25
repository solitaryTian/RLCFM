import torch
from diffusers import EulerAncestralDiscreteScheduler,FluxPipeline,UniPCMultistepScheduler,FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import os, json
from safetensors.torch import save_file
from pytorch_lightning import seed_everything
from tqdm import tqdm
from skrample.diffusers import SkrampleWrapperScheduler
from skrample.pytorch.noise import Brownian
from skrample.sampling import DPM
from skrample.scheduling import Beta




checkpoint_list = [4200]
device = 'cuda'

model_id = "/data/code/guanyu.zhao/models/FLUX.1-dev"
# sd_type = 'TDD_PCM_type_flux_niji_cfg1_proportion_empty_prompts_0_disable_bucket_Dev2Pro'
# sd_type = 'TDD_PCM_type_flux_niji_random_0_3_proportion_empty_prompts_0.2'
sd_type = 'TDD_PCM_type_flux_niji_cfg1_proportion_empty_prompts_0_disable_bucket'

lora_rank=64
weight_dtype = torch.float16
# resolution = 1024
seed = 100
seed_everything(seed)



validation_prompts1 = [
    '1girl, solo, ganyu (genshin impact)|||ahoge, artist name, bare shoulders, bell, black pantyhose, blue gloves, blue hair, blush, breasts, detached sleeves, gloves, gold trim, groin, horns, legs together, long hair, looking at viewer, medium breasts, neck bell, orb, pantyhose, parted lips, purple eyes, sideboob, sidelocks, skindentation, standing, tassel, thighlet, thighs, vision (genshin impact), white sleeves',
    '1girl, solo, manjuu (azur lane), noshiro (azur lane), noshiro (fragrance of the eastern snow) (azur lane)|||artist name, bare shoulders, black hair, china dress, chinese clothes, dress, feet, from side, full body, fur trim, gradient, horns, knees up, leg up, legs, long hair, looking at viewer, looking to the side, lying, no shoes, off shoulder, official alternate costume, on back, oni horns, open mouth, purple eyes, short dress, single thighhigh, sleeveless, sleeveless dress, smile, soles, thighhighs, thighs, toes, very long hair, white dress, white thighhighs',
    'anime, masterpiece, best quality, 1girl, solo, blush, sitting, twintails, blonde hair, bowtie, school uniform, nature',
    '1girl, full body,long hair, bangs, red hair, brown eyes, black blouse, short sleeve blouse,   blue sailor scarf, black hair clip, Wearing a sailor uniform and short skirt, enchanting body and a delicate face, anime style',
    '1girl, full body,long hair, bangs, red hair, brown eyes, black blouse, short sleeve blouse,   blue sailor scarf, black hair clip, Wearing a sailor uniform and short skirt, Laughing, reading a Comic Book, enchanting body and a delicate face, anime',
    '1girl, medium hair, bangs, curly hair, brown hair, brown eyes, white tanktops, off shoulder, turtleneck tanktops, pink skirt, long skirt, yellow earrings,silver necklace, orange high heels, campus cute girl , shiny eyes, elegant formal dress, miniskirt,  full body, symmetric, pixar style, minimalis, anime style',
    '1girl, very long hair, curly hair, black hair, black eyes, white dress, white hair clip, silver earrings, white veil, white gloves, A beautiful girl with a diamond crown, full body standing close-up, Asian face, exquisite appearance, moving expression, wearing a white luxury diamond wedding dress holding flowers, anime style',
    '1girl, long hair, bangs, curly hair, blue hair, green eyes, silver earrings,detailed Anime character portraits with underwater lighting, a Beautiful woman under water surrounded by fish, beautiful underwater scenes, anime style',
    'ultra-detailed,(best quality),((masterpiece)),(highres),original,extremely detailed 8K wallpaper,(an extremely delicate and beautiful), anime, BREAK 1girl,orange shoes,solo,sitting,sky,clouds,outdoors,black hair,bird,upward view,blue sky,white socks,daytime,orange jacket,building,long sleeves,leaves,long hair,stairs,red headband,pump Rope,headband,bangs,cloudy sky,from_below,wide_shot',
    '1girl, long hair, bangs, curly hair, yellow hair, yellow eyes, black dress, turtleneck dress, pink apron, purple hair ribbon, barbie anime showing off her athleticism and moves with her pole dancer skills',
    '1girl, very long hair, bangs, curly hair, brown hair, green eyes, purple shirt, black shorts, short shorts, pink hair ribbon, barbie anime showing off her athleticism and moves with her pole dancer skills',
    '1girl, very long hair, curl hair, black hair, black eyes, black dress, black hair clip, gold earrings, black veil, black gloves, A beautiful girl  a diamond crown,     full body standing close-up, Asian face,  wearing a black luxury diamond wedding dress, exquisite diamond jewelry, anime style',
    '1girl, very short hair, curl hair, blue hair, brown eyes, red dress, black hair clip, glod earrings, black veil, black gloves, A beautiful girl  a diamond crown,     full body standing close-up, Asian face,  wearing a black luxury diamond wedding dress,  exquisite diamond jewelry, anime style',
    '1girl, very short hair, curl hair, red hair, blue eyes, purple dress, pink hair clip, glod earrings, white veil, green gloves, A beautiful girl  a diamond crown,     full body standing close-up, Asian face,  wearing a black luxury diamond wedding dress,  exquisite diamond jewelry, anime style',
    '1girl, very long hair, straight hair, gray hair, green eyes, yellow dress, pink hair clip, gold earrings, black veil, orange gloves, A beautiful girl  a diamond crown,     full body standing close-up, Asian face,  wearing a black luxury diamond wedding dress,  exquisite diamond jewelry, anime style',
    '1girl, very long hair, straight hair, pink hair, red eyes, white dress, black hair clip, gold earrings, gray veil, violet gloves, A beautiful girl  a diamond crown,     full body standing close-up, Asian face,  wearing a black luxury diamond wedding dress,  exquisite diamond jewelry, anime style'
]
validation_prompts2 = [
'Masterpiece,best quality,ultra-detailed,detailed pupils,photography,high quality,top CG rendering,highest quality,ultra-clear,ultra-detailed,Official Art,Extremely Detailed CG,8k Wallpapers,HD,game CG,High quality,masterpiece,masterpiece,top CG rendering',
'A surreal photograph , a futuristic Dior latex pink overall with intricate liqyid metal details, jewels and mirrors. In a fancy futuristic room. Retro futuristic aesthetics. The scene is dark but beautiful, pink and cyan, golden. Pastel palette.',
'Colorful and simple background,  soft tones, while bright colors highlight their eyes. Soft colors, digital art style, cute illustrations for children"s book cover design, soft lighting, close-up shots, minimalist composition',
'A photograph. Created Using: vibrant colors, sharp focus, natural light, candid style, elegance, confidence, modern corporate aesthetic, glossy finish on the leather, glibatree prompt, high-resolution, digital photography',
'white gold metallic lines, roses, crescents, ribbons, diamonds, starlight, object floodlight, dreamy and beautiful atmosphere, delicate picture quality, high precision details, rendering, blender, object light, gorgeous, cinematic lighting, black background',
'photo-surrealism, hyper-realistic, cinematic, dramatic, vaporwave, japanese techwear, pale cyberpunk, moody, cloudy, thunder storm, dawn, foggy, reaching for the sky, pulled high into the sky above the wilderness',
'close-up, 4k, realistic',
'octane rendering, an screencap of an anime-style character. The character has large eyes and cute lips, smiling.  Hair has a flowing, watercolor-like texture, with shades, and it has a glossy finish. Decorations include  leaves and  flowers in hair. Expression appears friendly, with confidence and grace in eyes. The overall style is very delicate and refined, with a harmonious color palette.',
'3D render with a glowing circuit board pattern on its body, blue and teal lighting, in the style of cyberpunk, with glowing highlights and a bokeh background.',
'crafted from sea foam stands by the shore, with waves crashing and lashing against the rocks, only silhouette visible. The dress flows gracefully, seamlessly blending with the marine environment. gazes at the horizon under a dramatic sky. Sunlight accentuates the intricate details of the costume and the landscape. The gown is adorned with layers and textures reminiscent of seashells or coral, giving it depth and tactile quality. As stands under the sky painted by the setting sun, long shadows are cast across the wet sand, creating a dramatic contrast between light and shadow.',
'Thumbnail of a YouTube video. There is also a Japanese anime-style background of stars and space.',
'stunning and breathtaking wallpaper of norse shaman, amazing mystic shaman viking outfit with face painting and other head outfit, very expressive, amazing details, vivid colors, relaxing nordic atmosphere, perfect nature light for photography, amazing vibrant colors, impressive, cinematic photorealistic 8k',
'on a swing, made of white, transparent material, with a foggy background, in a fantasy style. A simple, clean picture, resembling a vintage oil painting.',
'XHMo,chenyu,hongchen,XH,Masterpiece,best quality,ultra-detailed,detailed pupils,photography,high quality,top CG rendering,highest quality,ultra-clear,ultra-detailed,Official Art,Extremely Detailed CG,8k Wallpapers,extreme aesthetic,in the style of fashion photography,light particles,cinematic lighting,visual impact,sharp focus,depth of field,dutch angle,HD,game CG,solo,hair ornament,long sleeves,closed mouth,upper body,flower,japanese clothes,indoors,hair flower,wide sleeves,kimono,sash,chinese clothes,own hands together,white flower,hanfu,',
'girl with perfect body,light gradient color long hair with bangs, light on cheeks,croptop, realistic skin texture, detailed picture, close-up, HD,32k，Her pupils sparkled with gradient color phantom stars,bright lighting atmosphere,highres,shinny skin, anime painting,the glowing letters "DAOTE" are prominently displayed',
'Luminous girl,GFSM,Luminous girl,GFSM,XS,Luminous girl,of a digital artwork depicting a serene,ethereal young woman with a mystical,otherworldly quality. The image is a highly detailed digital illustration in a fantasy style,showcasing a woman with a delicate,almost angelic appearance. She has fair skin and long,flowing,wavy hair that blends seamlessly into the night sky. Her hair is illuminated with various shades of blue and purple,creating a sparkling,cosmic effect that resembles stars and galaxies. Her eyes are closed,giving her a tranquil,meditative expression. Small,glowing,purple butterflies are delicately perched near her face,adding a sense of enchantment and magic to the scene. The woman’s lips are slightly parted,as if in a gentle,dreamy sigh. The background is a deep,dark blue,enhancing the ethereal and dreamlike atmosphere of the artwork. The overall texture is smooth and glossy,with a vibrant,luminescent quality that brings a sense of surrealism and enchantment to the piece.,',
'Digital artwork of a young woman with short,wavy blue hair and large blue eyes,wearing a light blue bra and sheer white robe,standing on a balcony overlooking the ocean. She has a fair complexion and a delicate,ethereal beauty. The background features a clear blue sky and calm sea.,',
'chenyu,hongchen,hongchen,Masterpiece,of the highest quality,ultra-detailed,with intricately detailed pupils,photography,top CG rendering,the utmost clarity,Official Art,Extremely Detailed CG,8k Wallpapers,good figure,HD,game CG,1girl,solo,long hair,breasts,looking at viewer,blue eyes,large breasts,black hair,hair ornament,animal ears,cleavage,sitting,collarbone,tail,full body,braid,parted lips,barefoot,indoors,hair flower,blurry,lips,see-through,fox ears,single braid,blurry background,fox tail,hair over shoulder,realistic,a girl is seated on a red carpet. She is dressed in a pink hanfu,adorned with flowers in her hair. Her hair is styled in a braided ponytail,adding a touch of color to her outfit. Her eyes are a piercing blue,and her eyebrows are a darker shade of brown. Her dress is adorned with a light pink floral pattern,while her hair is pulled back in a ponytail. The backdrop is a vibrant red,with a wooden table to the right of her,In this masterpiece of art,the viewer is treated to an ultra-high quality,intricately detailed scene. The photography and top CG rendering create a level of clarity and realism that is nothing short of breathtaking. Official Art and Extremely Detailed CG come together to produce 8k Wallpapers of the utmost quality.At the center of the piece is a solitary girl with long hair and breasts,her gaze fixed on the viewer. Her blue eyes are piercing,and her large breasts add to her allure. Her black hair is adorned with a hair ornament,and animal ears and a tail suggest a touch of the fantastical. She is seated,displaying her collarbone,and her full body is rendered with exquisite detail. A braid runs down her back,and her parted lips and bare feet give her a sense of vulnerability and openness.Her outfit is a pink hanfu,adorned with flowers in her hair and a braided ponytail that adds a touch of color to her ensemble. Her eyes are a piercing blue,set off by darker brown eyebrows. The dress is adorned with a l',
'Very pretty maiden,solo,Long hair, bust,Racing girl,Silver necklace, earrings,high heels,Translucent black pantyhose, Metal car paint material,integrated sheet metal with racing elements,giving it a misty,ghostly quality, Stand with your legs spread,Hands akimbo,dark blue background, realistic,High contrast colors,luminous palette,Natural movement,capture the moment,',
'Fashion-forward style: A model poses in a chic,monochromatic ensemble with black heels and stockings against an ethereal white backdrop.,',
'pingguoshouji:bizhi,Cyberpunk,shining,Old photographs that are badly damaged,Republic of China period,',
'chenyu,hongchen,hongchen,Masterpiece, of the highest quality, ultra-detailed, with intricately detailed pupils, photography, top CG rendering, the utmost clarity, Official Art, Extremely Detailed CG, 8k Wallpapers, good figure, HD, game CG,A beautiful woman cosplaying as Santa Claus, dressed in a form-fitting, woolen Christmas outfit with a touch of Chinese traditional style, stands in the center of a snowy winter wonderland. Snowflakes fall gently around her, illuminated by soft, warm lighting. She holds a large sack of gifts, smiling warmly as a team of reindeer pull a sleigh behind her. The scene is framed with the woman in the foreground, the sleigh and reindeer in the background, creating a magical and festive atmosphere with a unique blend of Western and Eastern aesthetics.',
'hsg,fur coat,A blonde Chinese hottie in white suspenders and a white fur coat,walking to show off her slender legs,silver sequined dress evening gown,blonde hair windy in the air,A stunning pin-up girl stands confidently on a bustling Tokyo street,head held high,eyes closed,exuding an air of effortless elegance. She wears a white-grey fur coat that perfectly complements her long legs and toned physique. Her O-ring lingerie adds a bold,modern edge to her look. A pair of chic black stockings enhances the seductive vibe,while the soft glow of night subtly highlights her features. Her dark hair cascades in loose waves,framing her face as she strikes a poised and alluring pose. The bokeh effect from the camera lens creates a dreamy halo around her,further accentuating her captivating beauty,fur coat,feather coat,Wearing a dress covered in big silver round sequins.,',
'pingguoshouji:bizhi,Cyberpunk,shining,phone wallpaper,iPhone wallpaper,',
'pingguoshouji:bizhi,Cyberpunk,shining,Old photographs that are badly damaged,Republic of China period,',
'fengge003,a lady with yellow light,sun,sunflower,narcissus,honey,sumac,front view,portrait,medium_shot,happy,',
'hsg,This is a clear full-body photo with a fashion-forward style. The person in the photo is a woman wearing a black lace-trimmed bra underneath a denim jacket,with jeans on below. Her hair is blonde and falls in waves over her shoulders. She is looking at the camera with a calm expression. She appears to be in her 20s or 30s,with a slim figure and delicate features. Her skin is fair,and her lips are tinted with a subtle shade of lipstick. Overall,the photo conveys a sense of fashion and confidence.,',
'Light Diffuser Photography,hsg,SIWA,XURE Chrome plated material,1girl,solo,long hair,skirt,simple background,jewelry,sitting,closed eyes,pantyhose,earrings,high heels,from side,blue skirt,black pantyhose,profile,blue background,crossed legs,high heel boots,skeleton,',
'realistic,Chrome plated material,High contrast colors,luminous palette,kawacy,strong emotional impact,highly detailed,dynamic,cinematic,stunning,realistic lighting and shading,vivid,vibrant,octane render,concept art,realistic,Cry engine,highly detailed,ultra-high resolution,32K UHD,sharp focus,best-quality,masterpiece,',
'Luminous cyberwave point light painted eye makeup,luminous lunar eyes,pure white luminous eye sockets,silvery white lunar eyeballs,pearlescent luminous eyebrow bow,silver long hair,rim light,fuzzy transparent skin shell,breathable and delicate makeup surface,holographic luminous ripple virtual clothing,dynamic composition,rim light,clean and dark purple gradient background,ultra-realistic futuristic silver tone,large depth of field,haziness,soft focus effect,light transmission extension effect beam,blurring the edge of the hair tail,the light painting effect,',
'hsg,1girl,tattoo,flower,solo,hair flower,hair ornament,back tattoo,earrings,back focus,(from behind:1.2),back,jewelry,white hair,red flower,blue eyes,upper body,looking back,white flower,looking at viewer,bare shoulders,blurry,indoors,short hair,rose,bangs,tassel,flower tattoo,off shoulder,tassel earrings,shoulder blades,nape,arm tattoo,red rose,blurry background,japanese clothes,neck tattoo,shoulder tattoo,bare back,closed mouth,kimono,medium hair,(close-up:1.4),indoor,east_asian_architecture,(night:1.3),(lighting:1.1),(from_below:1.4),(foreshortening:1.4),(Unusual composition:1.2),Back tattoo is the focus,',
'The laser card. The laser card is rectangular and has a colored black background with a gold to red gradation line around the card. The top left corner of the card is the "Genshin Impact" logo, It says "Xilonen" in gold letters. Below the logo, there is a gold trim with a floral pattern on it. In the center of the card, there is an illustration of the game character, xilonen \(genshin impact\), 1 girl, animal ears, cat ears, solo, green eyes, gradated hair, blonde hair, colored hair, hair ornaments, dark skinned female, dark skin, breasts, cleavage, medium breasts, jewelry, Bare shoulders, tail, tiger tail, ring, single glove, fur trim, bracelet, belly button, short shorts, blue shorts, denim, denim shorts. The illustration is cartoonish in style, with bright colors and exaggerated features.',
'ZOZ_SMAAA,1girl,1girl,The image depicts a person with short white hair, wearing a black dress with a deep neckline. They are seated at a piano, with their hands resting on the keys. The person is adorned with a gold choker and bracelet, and a gold chain is draped over their back. The piano is black with gold accents, and there is a reflection of the person in the piano"s lid. The background is dark, emphasizing the person and the piano. The overall scene has a luxurious and elegant feel, with a focus on the artistic and musical expression.,',
'WM,moon goddess sitting on an crescent moon,she is wearing long flowing dress with silver stars,Below the picture is water,in the style of anime.and the full-body portrait has an angelic appearance. The high-definition fantasy illustration is set against a black background.,painting,(oil painting:1.2),(oil painting strokes:1.1),',
'This image is a highly detailed, ultra-realistic CGI rendering of a young woman with a porcelain-white complexion and short platinum-blonde hair. She has a delicate, almost ethereal look, with large, expressive blue eyes and a delicate, slightly raised nose. Her lips are full and natural with a pink hue. She is dressed in a classic French maid outfit, consisting of an off-white ruffled headpiece with black trim, a black and white corset with a large bow in the middle, and a matching apron. The corset accentuates her small to medium-sized bust, and the fabric looks soft and slightly shiny, like a mix of satin and lace. She sits in a vintage, ornate chair with dark green upholstery and brass trim, adding a touch of elegance and antique charm to the scene. Her left hand is delicately placed beside her face, her fingers lightly touching her lips, while her right arm rests relaxedly on her knees. The background features a dark patterned wallpaper, which adds depth and a sense of vintage elegance to the image. The lighting is soft and diffuse, accentuating her features and the texture of the outfit and chair. The overall atmosphere is tranquil and refined. The light shines through the windows or plants on her face, casting complex patterns on her face',
'3d,cg,perfect clothes',
'1qqq,1qqq,Leica filters,Film photography,qrixi,Captured in a low-angle,eye-level perspective,a close-up shot of a young Asian woman with long dark hair,wearing a black blazer and a white t-shirt. Her lips are painted red,adding a pop of color to her face. Her eyes are a piercing blue,and her hair cascades over her shoulders. The backdrop is a stark blue,with white dots dotting the scene. The lighting is subdued,creating a stark contrast to the dark background. Cold tones,',
'illustration,A digital painting featuring a young boy with curly hair, glowing with vibrant hues of blue, orange, and green. He"s surrounded by colorful fish, swimming around his head. The background is a deep blue, creating a mystical underwater effect. The boy"s expression is serene and thoughtful, adding a dreamy, ethereal quality to the artwork.',
'1girl,solo,jewelry,hoop earrings,earrings,long hair,shorts,sitting,white footwear,crop top,shoes,denim shorts,sunglasses,breasts,denim,necklace,stool,off shoulder,sneakers,full body,black hair,tattoo,midriff,long sleeves,short shorts,bare legs,shirt,white shirt,round eyewear,bare shoulders,closed mouth,lips,cutoffs,shadow,between legs,tinted eyewear,hand between legs,chair,jacket,looking at viewer,tank top,open clothes,',
'kimono,',
'fantasy_witch,fantasy_witch A digital painting of a pale-skinned woman with long,wavy silver hair,wearing a black corset,white blouse,and thigh-high stockings. She sits on a stone ledge,her head adorned with a large,ornate hat. The background is a misty,gothic cityscape. The style is detailed and ethereal,with a slightly dark,mysterious atmosphere.,'
]


validation_prompts = validation_prompts1[:5] + validation_prompts2[:5]





# with open('/maindata/data/shared/public/songtao.tian/data/data_eval/aesthetic/prompts.json', 'r', encoding='utf-8') as f:
#     double_prompt_list = json.load(f)
# validation_prompts = double_prompt_list[::2]



steps_list = [4,8,16]
cfg_list = [3,4,5]
scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction",
    num_train_timesteps=1000,
)


lora_scale_list = [0.125]

for checkpoint in checkpoint_list:    
    pipeline = FluxPipeline.from_pretrained(model_id,torch_dtype=weight_dtype)
#     pipeline.scheduler = SkrampleWrapperScheduler(
#     sampler=DPM(order=2, add_noise=True, predictor=FLOW),
#     schedule=Beta(Flow(shift=3)),
# )

    b = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    pipeline.scheduler = SkrampleWrapperScheduler.from_diffusers_config(
        b
    )
    
    pipeline.set_progress_bar_config(disable=True)
    for lora_scale in lora_scale_list:
        lora_id = f'outputs/{sd_type}/checkpoint-{checkpoint}/pytorch_lora_weights.safetensors'
        pipeline.load_lora_weights(lora_id)
        pipeline.fuse_lora(lora_scale=lora_scale)
        pipeline.to(device)





        
        for steps in steps_list:
            for cfg in cfg_list:
                folder_path = f'outputs/{sd_type}/pictures/checkpoint{checkpoint}_step={steps},cfg={cfg},lora_scale={lora_scale}'
                # 检查文件夹是否存在
                if not os.path.exists(folder_path):
                    # 如果不存在，则创建文件夹
                    os.makedirs(folder_path)
                else:
                    print(f"文件夹 '{folder_path}' 已存在，无需创建。")

                for order, prompt in enumerate(tqdm(validation_prompts)):
                    image = pipeline(prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, height=1152, width=864).images[0]
                    image.save('{}/order={}.webp'.format(folder_path,order))

        pipeline.unload_lora_weights()
        pipeline.unfuse_lora()




