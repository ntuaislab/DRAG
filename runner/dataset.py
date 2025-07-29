"""
dataset.py
----------

Dataset module, provides dataset creation and pre-processors.
"""

import random
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets import (CocoCaptions, ImageFolder, ImageNet,
                                  VisionDataset)
from torchvision.transforms import v2
from transformers import BitImageProcessor, CLIPProcessor

from models.functional import UnNormalize

# preprocessor for openai/CLIP-ViT which retains gradient
CLIP_IMG_TRANSFORM = v2.Compose([
    v2.Resize(224, antialias=True),
    v2.CenterCrop(224),
    v2.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    ),
])

CLIP_IMG_UNNORMALIZE = UnNormalize(
    [0.48145466, 0.4578275, 0.40821073],
    [0.26862954, 0.26130258, 0.27577711]
)

# preprocessor for facebook/Dino which retains gradient
DINO_IMG_TRANSFORM = v2.Compose([
    v2.Resize(224, antialias=True),
    v2.CenterCrop(224),
    v2.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

DINO_IMG_UNNORMALIZE = UnNormalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

# preprocessor for latent diffusion model which retains gradient
LDM_IMG_TRANSFORM = v2.Compose([
    v2.Resize(512, interpolation=v2.InterpolationMode.BILINEAR),
    v2.CenterCrop(512),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

LDM_IMG_UNNORMALIZE = UnNormalize(
    [0.5],
    [0.5]
)

LSUN_CATEGORIES = [
    "bedroom",
    "bridge",
    "church_outdoor",
    "classroom",
    "conference_room",
    "dining_room",
    "kitchen",
    "living_room",
    "restaurant",
    "tower",
]

# Simpler human-readable labels for ImageNet
#
# Reference:
# https://github.com/anishathalye/imagenet-simple-labels
IMAGENET1K_CATEGORIES = [
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead shark",
    "electric ray",
    "stingray",
    "cock",
    "hen",
    "ostrich",
    "brambling",
    "goldfinch",
    "house finch",
    "junco",
    "indigo bunting",
    "American robin",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "American dipper",
    "kite",
    "bald eagle",
    "vulture",
    "great grey owl",
    "fire salamander",
    "smooth newt",
    "newt",
    "spotted salamander",
    "axolotl",
    "American bullfrog",
    "tree frog",
    "tailed frog",
    "loggerhead sea turtle",
    "leatherback sea turtle",
    "mud turtle",
    "terrapin",
    "box turtle",
    "banded gecko",
    "green iguana",
    "Carolina anole",
    "desert grassland whiptail lizard",
    "agama",
    "frilled-necked lizard",
    "alligator lizard",
    "Gila monster",
    "European green lizard",
    "chameleon",
    "Komodo dragon",
    "Nile crocodile",
    "American alligator",
    "triceratops",
    "worm snake",
    "ring-necked snake",
    "eastern hog-nosed snake",
    "smooth green snake",
    "kingsnake",
    "garter snake",
    "water snake",
    "vine snake",
    "night snake",
    "boa constrictor",
    "African rock python",
    "Indian cobra",
    "green mamba",
    "sea snake",
    "Saharan horned viper",
    "eastern diamondback rattlesnake",
    "sidewinder",
    "trilobite",
    "harvestman",
    "scorpion",
    "yellow garden spider",
    "barn spider",
    "European garden spider",
    "southern black widow",
    "tarantula",
    "wolf spider",
    "tick",
    "centipede",
    "black grouse",
    "ptarmigan",
    "ruffed grouse",
    "prairie grouse",
    "peacock",
    "quail",
    "partridge",
    "grey parrot",
    "macaw",
    "sulphur-crested cockatoo",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "duck",
    "red-breasted merganser",
    "goose",
    "black swan",
    "tusker",
    "echidna",
    "platypus",
    "wallaby",
    "koala",
    "wombat",
    "jellyfish",
    "sea anemone",
    "brain coral",
    "flatworm",
    "nematode",
    "conch",
    "snail",
    "slug",
    "sea slug",
    "chiton",
    "chambered nautilus",
    "Dungeness crab",
    "rock crab",
    "fiddler crab",
    "red king crab",
    "American lobster",
    "spiny lobster",
    "crayfish",
    "hermit crab",
    "isopod",
    "white stork",
    "black stork",
    "spoonbill",
    "flamingo",
    "little blue heron",
    "great egret",
    "bittern",
    "crane (bird)",
    "limpkin",
    "common gallinule",
    "American coot",
    "bustard",
    "ruddy turnstone",
    "dunlin",
    "common redshank",
    "dowitcher",
    "oystercatcher",
    "pelican",
    "king penguin",
    "albatross",
    "grey whale",
    "killer whale",
    "dugong",
    "sea lion",
    "Chihuahua",
    "Japanese Chin",
    "Maltese",
    "Pekingese",
    "Shih Tzu",
    "King Charles Spaniel",
    "Papillon",
    "toy terrier",
    "Rhodesian Ridgeback",
    "Afghan Hound",
    "Basset Hound",
    "Beagle",
    "Bloodhound",
    "Bluetick Coonhound",
    "Black and Tan Coonhound",
    "Treeing Walker Coonhound",
    "English foxhound",
    "Redbone Coonhound",
    "borzoi",
    "Irish Wolfhound",
    "Italian Greyhound",
    "Whippet",
    "Ibizan Hound",
    "Norwegian Elkhound",
    "Otterhound",
    "Saluki",
    "Scottish Deerhound",
    "Weimaraner",
    "Staffordshire Bull Terrier",
    "American Staffordshire Terrier",
    "Bedlington Terrier",
    "Border Terrier",
    "Kerry Blue Terrier",
    "Irish Terrier",
    "Norfolk Terrier",
    "Norwich Terrier",
    "Yorkshire Terrier",
    "Wire Fox Terrier",
    "Lakeland Terrier",
    "Sealyham Terrier",
    "Airedale Terrier",
    "Cairn Terrier",
    "Australian Terrier",
    "Dandie Dinmont Terrier",
    "Boston Terrier",
    "Miniature Schnauzer",
    "Giant Schnauzer",
    "Standard Schnauzer",
    "Scottish Terrier",
    "Tibetan Terrier",
    "Australian Silky Terrier",
    "Soft-coated Wheaten Terrier",
    "West Highland White Terrier",
    "Lhasa Apso",
    "Flat-Coated Retriever",
    "Curly-coated Retriever",
    "Golden Retriever",
    "Labrador Retriever",
    "Chesapeake Bay Retriever",
    "German Shorthaired Pointer",
    "Vizsla",
    "English Setter",
    "Irish Setter",
    "Gordon Setter",
    "Brittany Spaniel",
    "Clumber Spaniel",
    "English Springer Spaniel",
    "Welsh Springer Spaniel",
    "Cocker Spaniels",
    "Sussex Spaniel",
    "Irish Water Spaniel",
    "Kuvasz",
    "Schipperke",
    "Groenendael",
    "Malinois",
    "Briard",
    "Australian Kelpie",
    "Komondor",
    "Old English Sheepdog",
    "Shetland Sheepdog",
    "collie",
    "Border Collie",
    "Bouvier des Flandres",
    "Rottweiler",
    "German Shepherd Dog",
    "Dobermann",
    "Miniature Pinscher",
    "Greater Swiss Mountain Dog",
    "Bernese Mountain Dog",
    "Appenzeller Sennenhund",
    "Entlebucher Sennenhund",
    "Boxer",
    "Bullmastiff",
    "Tibetan Mastiff",
    "French Bulldog",
    "Great Dane",
    "St. Bernard",
    "husky",
    "Alaskan Malamute",
    "Siberian Husky",
    "Dalmatian",
    "Affenpinscher",
    "Basenji",
    "pug",
    "Leonberger",
    "Newfoundland",
    "Pyrenean Mountain Dog",
    "Samoyed",
    "Pomeranian",
    "Chow Chow",
    "Keeshond",
    "Griffon Bruxellois",
    "Pembroke Welsh Corgi",
    "Cardigan Welsh Corgi",
    "Toy Poodle",
    "Miniature Poodle",
    "Standard Poodle",
    "Mexican hairless dog",
    "grey wolf",
    "Alaskan tundra wolf",
    "red wolf",
    "coyote",
    "dingo",
    "dhole",
    "African wild dog",
    "hyena",
    "red fox",
    "kit fox",
    "Arctic fox",
    "grey fox",
    "tabby cat",
    "tiger cat",
    "Persian cat",
    "Siamese cat",
    "Egyptian Mau",
    "cougar",
    "lynx",
    "leopard",
    "snow leopard",
    "jaguar",
    "lion",
    "tiger",
    "cheetah",
    "brown bear",
    "American black bear",
    "polar bear",
    "sloth bear",
    "mongoose",
    "meerkat",
    "tiger beetle",
    "ladybug",
    "ground beetle",
    "longhorn beetle",
    "leaf beetle",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "ant",
    "grasshopper",
    "cricket",
    "stick insect",
    "cockroach",
    "mantis",
    "cicada",
    "leafhopper",
    "lacewing",
    "dragonfly",
    "damselfly",
    "red admiral",
    "ringlet",
    "monarch butterfly",
    "small white",
    "sulphur butterfly",
    "gossamer-winged butterfly",
    "starfish",
    "sea urchin",
    "sea cucumber",
    "cottontail rabbit",
    "hare",
    "Angora rabbit",
    "hamster",
    "porcupine",
    "fox squirrel",
    "marmot",
    "beaver",
    "guinea pig",
    "common sorrel",
    "zebra",
    "pig",
    "wild boar",
    "warthog",
    "hippopotamus",
    "ox",
    "water buffalo",
    "bison",
    "ram",
    "bighorn sheep",
    "Alpine ibex",
    "hartebeest",
    "impala",
    "gazelle",
    "dromedary",
    "llama",
    "weasel",
    "mink",
    "European polecat",
    "black-footed ferret",
    "otter",
    "skunk",
    "badger",
    "armadillo",
    "three-toed sloth",
    "orangutan",
    "gorilla",
    "chimpanzee",
    "gibbon",
    "siamang",
    "guenon",
    "patas monkey",
    "baboon",
    "macaque",
    "langur",
    "black-and-white colobus",
    "proboscis monkey",
    "marmoset",
    "white-headed capuchin",
    "howler monkey",
    "titi",
    "Geoffroy's spider monkey",
    "common squirrel monkey",
    "ring-tailed lemur",
    "indri",
    "Asian elephant",
    "African bush elephant",
    "red panda",
    "giant panda",
    "snoek",
    "eel",
    "coho salmon",
    "rock beauty",
    "clownfish",
    "sturgeon",
    "garfish",
    "lionfish",
    "pufferfish",
    "abacus",
    "abaya",
    "academic gown",
    "accordion",
    "acoustic guitar",
    "aircraft carrier",
    "airliner",
    "airship",
    "altar",
    "ambulance",
    "amphibious vehicle",
    "analog clock",
    "apiary",
    "apron",
    "waste container",
    "assault rifle",
    "backpack",
    "bakery",
    "balance beam",
    "balloon",
    "ballpoint pen",
    "Band-Aid",
    "banjo",
    "baluster",
    "barbell",
    "barber chair",
    "barbershop",
    "barn",
    "barometer",
    "barrel",
    "wheelbarrow",
    "baseball",
    "basketball",
    "bassinet",
    "bassoon",
    "swimming cap",
    "bath towel",
    "bathtub",
    "station wagon",
    "lighthouse",
    "beaker",
    "military cap",
    "beer bottle",
    "beer glass",
    "bell-cot",
    "bib",
    "tandem bicycle",
    "bikini",
    "ring binder",
    "binoculars",
    "birdhouse",
    "boathouse",
    "bobsleigh",
    "bolo tie",
    "poke bonnet",
    "bookcase",
    "bookstore",
    "bottle cap",
    "bow",
    "bow tie",
    "brass",
    "bra",
    "breakwater",
    "breastplate",
    "broom",
    "bucket",
    "buckle",
    "bulletproof vest",
    "high-speed train",
    "butcher shop",
    "taxicab",
    "cauldron",
    "candle",
    "cannon",
    "canoe",
    "can opener",
    "cardigan",
    "car mirror",
    "carousel",
    "tool kit",
    "carton",
    "car wheel",
    "automated teller machine",
    "cassette",
    "cassette player",
    "castle",
    "catamaran",
    "CD player",
    "cello",
    "mobile phone",
    "chain",
    "chain-link fence",
    "chain mail",
    "chainsaw",
    "chest",
    "chiffonier",
    "chime",
    "china cabinet",
    "Christmas stocking",
    "church",
    "movie theater",
    "cleaver",
    "cliff dwelling",
    "cloak",
    "clogs",
    "cocktail shaker",
    "coffee mug",
    "coffeemaker",
    "coil",
    "combination lock",
    "computer keyboard",
    "confectionery store",
    "container ship",
    "convertible",
    "corkscrew",
    "cornet",
    "cowboy boot",
    "cowboy hat",
    "cradle",
    "crane (machine)",
    "crash helmet",
    "crate",
    "infant bed",
    "Crock Pot",
    "croquet ball",
    "crutch",
    "cuirass",
    "dam",
    "desk",
    "desktop computer",
    "rotary dial telephone",
    "diaper",
    "digital clock",
    "digital watch",
    "dining table",
    "dishcloth",
    "dishwasher",
    "disc brake",
    "dock",
    "dog sled",
    "dome",
    "doormat",
    "drilling rig",
    "drum",
    "drumstick",
    "dumbbell",
    "Dutch oven",
    "electric fan",
    "electric guitar",
    "electric locomotive",
    "entertainment center",
    "envelope",
    "espresso machine",
    "face powder",
    "feather boa",
    "filing cabinet",
    "fireboat",
    "fire engine",
    "fire screen sheet",
    "flagpole",
    "flute",
    "folding chair",
    "football helmet",
    "forklift",
    "fountain",
    "fountain pen",
    "four-poster bed",
    "freight car",
    "French horn",
    "frying pan",
    "fur coat",
    "garbage truck",
    "gas mask",
    "gas pump",
    "goblet",
    "go-kart",
    "golf ball",
    "golf cart",
    "gondola",
    "gong",
    "gown",
    "grand piano",
    "greenhouse",
    "grille",
    "grocery store",
    "guillotine",
    "barrette",
    "hair spray",
    "half-track",
    "hammer",
    "hamper",
    "hair dryer",
    "hand-held computer",
    "handkerchief",
    "hard disk drive",
    "harmonica",
    "harp",
    "harvester",
    "hatchet",
    "holster",
    "home theater",
    "honeycomb",
    "hook",
    "hoop skirt",
    "horizontal bar",
    "horse-drawn vehicle",
    "hourglass",
    "iPod",
    "clothes iron",
    "jack-o'-lantern",
    "jeans",
    "jeep",
    "T-shirt",
    "jigsaw puzzle",
    "pulled rickshaw",
    "joystick",
    "kimono",
    "knee pad",
    "knot",
    "lab coat",
    "ladle",
    "lampshade",
    "laptop computer",
    "lawn mower",
    "lens cap",
    "paper knife",
    "library",
    "lifeboat",
    "lighter",
    "limousine",
    "ocean liner",
    "lipstick",
    "slip-on shoe",
    "lotion",
    "speaker",
    "loupe",
    "sawmill",
    "magnetic compass",
    "mail bag",
    "mailbox",
    "tights",
    "tank suit",
    "manhole cover",
    "maraca",
    "marimba",
    "mask",
    "match",
    "maypole",
    "maze",
    "measuring cup",
    "medicine chest",
    "megalith",
    "microphone",
    "microwave oven",
    "military uniform",
    "milk can",
    "minibus",
    "miniskirt",
    "minivan",
    "missile",
    "mitten",
    "mixing bowl",
    "mobile home",
    "Model T",
    "modem",
    "monastery",
    "monitor",
    "moped",
    "mortar",
    "square academic cap",
    "mosque",
    "mosquito net",
    "scooter",
    "mountain bike",
    "tent",
    "computer mouse",
    "mousetrap",
    "moving van",
    "muzzle",
    "nail",
    "neck brace",
    "necklace",
    "nipple",
    "notebook computer",
    "obelisk",
    "oboe",
    "ocarina",
    "odometer",
    "oil filter",
    "organ",
    "oscilloscope",
    "overskirt",
    "bullock cart",
    "oxygen mask",
    "packet",
    "paddle",
    "paddle wheel",
    "padlock",
    "paintbrush",
    "pajamas",
    "palace",
    "pan flute",
    "paper towel",
    "parachute",
    "parallel bars",
    "park bench",
    "parking meter",
    "passenger car",
    "patio",
    "payphone",
    "pedestal",
    "pencil case",
    "pencil sharpener",
    "perfume",
    "Petri dish",
    "photocopier",
    "plectrum",
    "Pickelhaube",
    "picket fence",
    "pickup truck",
    "pier",
    "piggy bank",
    "pill bottle",
    "pillow",
    "ping-pong ball",
    "pinwheel",
    "pirate ship",
    "pitcher",
    "hand plane",
    "planetarium",
    "plastic bag",
    "plate rack",
    "plow",
    "plunger",
    "Polaroid camera",
    "pole",
    "police van",
    "poncho",
    "billiard table",
    "soda bottle",
    "pot",
    "potter's wheel",
    "power drill",
    "prayer rug",
    "printer",
    "prison",
    "projectile",
    "projector",
    "hockey puck",
    "punching bag",
    "purse",
    "quill",
    "quilt",
    "race car",
    "racket",
    "radiator",
    "radio",
    "radio telescope",
    "rain barrel",
    "recreational vehicle",
    "reel",
    "reflex camera",
    "refrigerator",
    "remote control",
    "restaurant",
    "revolver",
    "rifle",
    "rocking chair",
    "rotisserie",
    "eraser",
    "rugby ball",
    "ruler",
    "running shoe",
    "safe",
    "safety pin",
    "salt shaker",
    "sandal",
    "sarong",
    "saxophone",
    "scabbard",
    "weighing scale",
    "school bus",
    "schooner",
    "scoreboard",
    "CRT screen",
    "screw",
    "screwdriver",
    "seat belt",
    "sewing machine",
    "shield",
    "shoe store",
    "shoji",
    "shopping basket",
    "shopping cart",
    "shovel",
    "shower cap",
    "shower curtain",
    "ski",
    "ski mask",
    "sleeping bag",
    "slide rule",
    "sliding door",
    "slot machine",
    "snorkel",
    "snowmobile",
    "snowplow",
    "soap dispenser",
    "soccer ball",
    "sock",
    "solar thermal collector",
    "sombrero",
    "soup bowl",
    "space bar",
    "space heater",
    "space shuttle",
    "spatula",
    "motorboat",
    "spider web",
    "spindle",
    "sports car",
    "spotlight",
    "stage",
    "steam locomotive",
    "through arch bridge",
    "steel drum",
    "stethoscope",
    "scarf",
    "stone wall",
    "stopwatch",
    "stove",
    "strainer",
    "tram",
    "stretcher",
    "couch",
    "stupa",
    "submarine",
    "suit",
    "sundial",
    "sunglass",
    "sunglasses",
    "sunscreen",
    "suspension bridge",
    "mop",
    "sweatshirt",
    "swimsuit",
    "swing",
    "switch",
    "syringe",
    "table lamp",
    "tank",
    "tape player",
    "teapot",
    "teddy bear",
    "television",
    "tennis ball",
    "thatched roof",
    "front curtain",
    "thimble",
    "threshing machine",
    "throne",
    "tile roof",
    "toaster",
    "tobacco shop",
    "toilet seat",
    "torch",
    "totem pole",
    "tow truck",
    "toy store",
    "tractor",
    "semi-trailer truck",
    "tray",
    "trench coat",
    "tricycle",
    "trimaran",
    "tripod",
    "triumphal arch",
    "trolleybus",
    "trombone",
    "tub",
    "turnstile",
    "typewriter keyboard",
    "umbrella",
    "unicycle",
    "upright piano",
    "vacuum cleaner",
    "vase",
    "vault",
    "velvet",
    "vending machine",
    "vestment",
    "viaduct",
    "violin",
    "volleyball",
    "waffle iron",
    "wall clock",
    "wallet",
    "wardrobe",
    "military aircraft",
    "sink",
    "washing machine",
    "water bottle",
    "water jug",
    "water tower",
    "whiskey jug",
    "whistle",
    "wig",
    "window screen",
    "window shade",
    "Windsor tie",
    "wine bottle",
    "wing",
    "wok",
    "wooden spoon",
    "wool",
    "split-rail fence",
    "shipwreck",
    "yawl",
    "yurt",
    "website",
    "comic book",
    "crossword",
    "traffic sign",
    "traffic light",
    "dust jacket",
    "menu",
    "plate",
    "guacamole",
    "consomme",
    "hot pot",
    "trifle",
    "ice cream",
    "ice pop",
    "baguette",
    "bagel",
    "pretzel",
    "cheeseburger",
    "hot dog",
    "mashed potato",
    "cabbage",
    "broccoli",
    "cauliflower",
    "zucchini",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber",
    "artichoke",
    "bell pepper",
    "cardoon",
    "mushroom",
    "Granny Smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard apple",
    "pomegranate",
    "hay",
    "carbonara",
    "chocolate syrup",
    "dough",
    "meatloaf",
    "pizza",
    "pot pie",
    "burrito",
    "red wine",
    "espresso",
    "cup",
    "eggnog",
    "alp",
    "bubble",
    "cliff",
    "coral reef",
    "geyser",
    "lakeshore",
    "promontory",
    "shoal",
    "seashore",
    "valley",
    "volcano",
    "baseball player",
    "bridegroom",
    "scuba diver",
    "rapeseed",
    "daisy",
    "yellow lady's slipper",
    "corn",
    "acorn",
    "rose hip",
    "horse chestnut seed",
    "coral fungus",
    "agaric",
    "gyromitra",
    "stinkhorn mushroom",
    "earth star",
    "hen-of-the-woods",
    "bolete",
    "ear of corn",
    "toilet paper"
]


def create_normalizer_and_unnormalizer(
    name: str
) -> Tuple[nn.Module, nn.Module]:
    """ Create normalizer and unnormalizer.

    Arguments
    ---------
    name : str
        Normalizer name.

    Returns
    -------
    Tuple[nn.Module, nn.Module]
        Normalizer and unnormalizer.
    """
    match name:
        case 'ModifiedResNet' | 'CLIPVisionModelWithProjection':
            return CLIP_IMG_TRANSFORM, CLIP_IMG_UNNORMALIZE
        case 'Dinov2Model':
            return DINO_IMG_TRANSFORM, DINO_IMG_UNNORMALIZE
        case 'ImageGenerator' | 'InpaintingOperator':
            return image_generator(), LDM_IMG_UNNORMALIZE
        case _:
            raise ValueError(f'Unknown normalizer: {name}')


def clip_vit_processor(**kwargs):
    """ Return CLIP processor.

    Returns
    -------
    Callable
        CLIP image and text preprocessor.

    Raises
    ------
    ValueError
        If kwargs is not None.
    """
    if kwargs:
        raise ValueError('Does not support kwargs.')

    # CLIP Processor is not provided in openai/clip-vit-large-patch14
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    return lambda x: processor(images=x, return_tensors='pt', padding=True).pixel_values.squeeze(0)


def dinov2_processor(**kwargs):
    """ Return DINOv2 processor.

    Returns
    -------
    Callable
        DINOv2 image preprocessor.

    Raises
    ------
    ValueError
        If kwargs is not None.
    """
    if kwargs:
        raise ValueError('Does not support kwargs.')

    processor = BitImageProcessor.from_pretrained('facebook/dinov2-base')

    # Hack: the processor, by default, first resize the image to 256, then crop center to 224.
    # We want to keep it aligned with CLIP-ViT preprocessor, which don't crop out the edges.
    processor.size['shortest_edge'] = 224

    # unpack batch: assumed that the preprocessor are used by dataloader,
    # which preparing data w/ many workers.
    return lambda x: processor(images=x, return_tensors='pt').pixel_values[0]


def ldm_processor(**kwargs):
    """ Return LDM processor.

    Returns
    -------
    Callable
        LDM image preprocessor.

    Raises
    ------
    ValueError
        If kwargs is not None.
    """
    res = kwargs.pop('resolution', 512)
    assert isinstance(res, int) and res > 0, f'Invalid resolution: {res}'

    return v2.Compose([
        v2.Resize(res, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(res),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])


def image_generator(**kwargs):
    """ Return LDM processor.

    Returns
    -------
    Callable
        LDM image preprocessor.

    Raises
    ------
    ValueError
        If kwargs is not None.
    """
    res = kwargs.pop('resolution', 256)
    assert isinstance(res, int) and res > 0, f'Invalid resolution: {res}'

    return v2.Compose([
        v2.Resize(res, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(res),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])


def create_transforms(
    transform_config: DictConfig | Dict | None
) -> Tuple[Callable, Callable | None]:
    """ Create transform from config.

    Arguments
    ---------
    transform : DictConfig | None
        Transform config.

    Returns
    -------
    Tuple[Callable, Callable | None]
        Input and target transform functions.
    """
    strmap = {
        'auto_augment': lambda: v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
        'center_crop': v2.CenterCrop,
        'gaussian_blur': v2.GaussianBlur,
        'normalize': v2.Normalize,
        'resize': v2.Resize,
        'random_horizontal_flip': v2.RandomHorizontalFlip,
        'random_erasing': v2.RandomErasing,
        'to_tensor': v2.ToTensor,
        'to_image': v2.ToImage,
        'to_dtype': v2.ToDtype,
        'clip_vit_processor': clip_vit_processor,
        'dinov2_processor': dinov2_processor,
        'image_generator': image_generator,
        'ldm_processor': ldm_processor,
    }

    if transform_config is None:
        return (None, None)

    if isinstance(transform_config, DictConfig):
        transform_config = OmegaConf.to_object(transform_config)

    transforms = []
    for transform in transform_config:
        if len(transform.keys()) != 1:
            raise ValueError(f'Unknown transform: {transform}')

        name = list(transform.keys())[0]
        args = transform[name]

        if isinstance(args, (DictConfig, ListConfig)):
            args = OmegaConf.to_object(args)

        try:
            transforms.append(strmap[name](**args))
        except Exception as err:
            print(f'Error: {name}, {args}') # Show error message for debugging
            raise err

    return (v2.Compose(transforms), None)


def mscoco_collate_fn(batch):
    """ Collate function for COCO caption dataset.

    This function selects a random caption from the list of captions for each image.
    """
    # Unzip the batch
    images, captions = zip(*batch)

    # Concatenate the images along the batch dimension
    images = torch.stack(images, dim=0)

    # Flatten the list of captions and keep track of lengths
    captions = [random.choice(sublist) for sublist in captions]
    # lengths = torch.tensor([len(caption) for caption in flat_captions])

    # Pad the captions to the maximum length in this batch
    # padded_captions = pad_sequence(flat_captions, batch_first=True, padding_value=0)

    return images, captions # padded_captions, lengths


class FFHQ(VisionDataset):
    """ FFHQ dataset class. """

    filenames: List[Path]

    def __init__(
        self,
        root: PathLike,
        resolution: int = 128,
        split: str = 'train',
        transform: Callable | None = None,
        target_transform: Callable | None = None
    ) -> None:
        """
        Arguments
        ---------
        root : PathLike
            Root directory of dataset.

        resolution : int
            Resolution of images. Choices: {128, 1024}. Default: 128.

        transform : Callable | None
            Image transform function.
        """
        root = Path(root)

        match resolution:
            case 128:
                root = root / 'thumbnails128x128'
            case 1024:
                root = root / 'images1024x1024'
            case _:
                raise ValueError(f'Invalid resolution: {resolution}')

        super().__init__(root, transform=transform)

        # Folder hierarchy
        # root
        # ├── 00000
        # ├── 01000
        # ├── ...
        # ├── 68000
        # ├── 69000
        # └── LICENSE.txt
        match split:
            case 'train':
                start, end = 0, 60000
            case 'val':
                start, end = 60000, 70000
            case 'trainval':
                start, end = 0, 70000
            case _:
                raise ValueError(f'Invalid split: {split}')

        filenames: List[Path] = []
        for grp in range(start, end, 1000):
            filenames.extend(sorted((root / str(grp).zfill(5)).iterdir()))

        self.filenames = filenames

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        image = self._load_image(idx)
        if self.transform is not None:
            image = self.transform(image)
        return image, idx

    def _load_image(self, idx: int) -> Any:
        return Image.open(self.filenames[idx]).convert('RGB')

    def __len__(self) -> int:
        return len(self.filenames)


def create_dataset(
    root: str,
    name: str,
    split: str,
    **kwargs
) -> Tuple[Dataset, Callable | None]:
    """ Return dataset constructor given class name.

    Arguments
    ---------
    root : str
        Root directory of dataset.

    name : str
        Name of dataset.

    split : str
        Split of dataset.

    Returns
    -------
    Dataset
        Dataset instance.

    Callable | None
        Collate function.

    Raises
    ------
    ValueError
        If dataset name is not recognized.
    """
    if name == 'mscoco':
        if (year := kwargs.pop('year', None)) is None:
            raise ValueError('Year is required for COCOCaption dataset')

        split = 'val' if split == 'valid' else split
        return (CocoCaptions(
            root=Path(root) / 'coco' / 'images' / f'{split}{year}',
            annFile=Path(root) / 'coco' / 'annotations' / f'captions_{split}{year}.json', # type: ignore
            **kwargs
        ), mscoco_collate_fn)

    if name == 'ffhq':
        split = 'val' if split == 'valid' else split
        return (FFHQ(Path(root) / 'ffhq', 128, split, **kwargs), None)

    if name in ('imagenet', 'imagenet2012'):
        split = 'val' if split == 'valid' else split
        return (ImageNet(Path(root) / 'imagenet2012', split, **kwargs), None)

    if name == 'UCMerced_LandUse':
        return (
            ImageFolder(Path(root) / 'UCMerced_LandUse' / 'Images', **kwargs), # type: ignore
            None
        )

    raise ValueError(f'Unknown dataset class: {name}')


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, pair: Any, max_steps: int = 1):
        super().__init__()

        self.pair = pair
        self.max_steps = max_steps

    def __len__(self):
        return self.max_steps

    def __iter__(self):
        for _ in range(self.max_steps):
            yield self.pair
