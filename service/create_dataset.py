import os
import shutil
from PIL import Image
from zipfile import ZipFile
from tqdm import tqdm
import random

NUMBER_OF_EXAMPLES = 500
SHUFFLE = True
IMAGE_SIZE = (180, 200)

# categories_to_train = ["MMX00007","MMX00008", "MMX00009","MMX00010"]
product_description = {x.strip().split("_")[0]: x.strip().split("_")[1] for x in
                       """MMX00001_Ferrero Rocher T16;MMX00002_Milka caramel chocolate;MMX00003_Kinder Chocolate Classic;MMX00004_Twix Xtra;MMX00005_Toblerone;MMX00006_Toblerone Bitter;MMX00007_Lays Classic;MMX00008_Lays Chilly;MMX00009_Pringles Original;MMX00010_Pringles Paprika;MMX00011_Nestle Purelife Water;MMX00012_San Pellegrino Mineral Water;MMX00013_Red bull 250;MMX00014_Monster Energy Drink;MMX00015_Heinz ketchup;MMX00016_Heinz Mayonnaise;MMX00017_Barilla Pesto;MMX00018_Barilla Sugo;MMX00019_Penne Barilla;MMX00020_Lasagne Barilla;MMX00021_L'Oréal Paris Elsève Color-Vive Shampoo;MMX00022_Dove Beauty Cream Bar;MMX00023_Sensodyne Repair & Protect;MMX00024_Colgate Triple Action;MMX00025_Sensodyne Cool Mint Mouthwash;MMX00026_Nivea men cool kick rollon;MMX00027_Rexona sexy bouqet sprey deodorant women;MMX00028_Dove original sprey deodorant women;MMX00029_Nivea Baby Smoothy Cream;MMX00030_Johnson's baby oil;""".split(
                           ";") if x != ""}

if "finetuned" not in os.listdir("../"):
    os.mkdir("../finetuned")
if "dataset" in os.listdir("../finetuned"):
    shutil.rmtree("../finetuned/dataset", ignore_errors=True)
if "dataset" not in os.listdir("../finetuned"):
    os.mkdir("../finetuned/dataset")

FOLDER_PATH = "../products"
if "products" not in os.listdir():
    os.mkdir(FOLDER_PATH)
if "ProductDataset_30" not in os.listdir(FOLDER_PATH):
    with ZipFile("./ProductDataset_30.zip", 'r') as zObject:
        zObject.extractall(path=FOLDER_PATH)

n = 0
for cat in tqdm(os.listdir("../products/ProductDataset_30"), desc="Creating dataset for categories"):
    # if cat not in categories_to_train:
    #  continue
    category_files = os.listdir(f"products/ProductDataset_30/{cat}")
    if SHUFFLE:
        random.shuffle(category_files)
    for n_category, product in enumerate(category_files):
        if n_category > NUMBER_OF_EXAMPLES:
            continue
        Image.open(f"products/ProductDataset_30/{cat}/{product}").resize(IMAGE_SIZE).save(f"finetuned/dataset/{n}.png")
        with open(f"finetuned/dataset/{n}.txt", "w") as file:
            file.write(product_description[cat])
        n += 1
