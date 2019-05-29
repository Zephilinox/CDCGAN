import os
import cv2

data = "./data"
unpacked = "./unpacked"
blocks_path = "assets/minecraft/textures/blocks"

unpacked_folders = os.listdir(unpacked)

exportable = [
    "dirt",
    "sand",
    "snow",
    "stone",
    "cobblestone",
    "grass_top",
    "grass_side",
    "gravel",
    "ice",
    "log_birch_top",
    "log_birch",
    "log_oak_top",
    "log_oak",
    "quartz_ore",
    "redstone_ore",
    "coal_ore",
    "diamond_ore",
    "emerald_ore",
    "gold_ore",
    "iron_ore",
    "lapis_ore",
]

rename = [
    "dirt",
    "sand",
    "snow",
    "stone",
    "cobblestone",
    "grass_block_top",
    "grass_block_side",
    "gravel",
    "ice",
    "birch_log_top",
    "birch_log",
    "oak_log_top",
    "oak_log",
    "quartz_ore",
    "redstone_ore",
    "coal_ore",
    "diamond_ore",
    "emerald_ore",
    "gold_ore",
    "iron_ore",
    "lapis_ore",
]

def extractBlocksFromUnpackedTexturePack():
    print("hi")
    if True:
        print("hi")
        
    for folder in unpacked_folders:
        print(folder)
        if os.path.isdir(unpacked + "/" + folder + "/" + blocks_path):
            textures = os.listdir(unpacked + "/" + folder + "/" + blocks_path)		
            for texture in textures:
                texture_is_exportable = texture[:-4] in exportable
                texture_needs_renaming = texture[:-4] in rename

                if texture[-4:] == ".png" and (texture_is_exportable or texture_needs_renaming):
                    source = unpacked + "/" + folder + "/" + blocks_path + "/" + texture
                    destination = data + "/" + texture[:-4] + "/"
                    newName = folder + ".png"
                    os.makedirs(destination, exist_ok=True)
                    print("copying " + texture + " to " + destination + newName)
                    img = cv2.imread(source)
                    img = cv2.resize(img,(64,64), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(destination + newName, img)
	
extractBlocksFromUnpackedTexturePack()