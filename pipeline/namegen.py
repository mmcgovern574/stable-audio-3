"""Novel-name generator for the catalog.

Only renames DUPLICATES: for each title word, the best-rated loop keeps the word; every other
loop that reused that word gets a fresh, on-brand evocative word. Deterministic (sorted by id),
so re-runs are stable. Single words in the cKz dark/melodic aesthetic — no template, no suffix.
"""
from collections import defaultdict

# Curated, on-brand single words (dark / cosmic / luxe / mystic). Plenty of headroom.
POOL = [
    "Eclipse", "Phantom", "Venom", "Cobra", "Sable", "Raven", "Banshee", "Reaper", "Hollow",
    "Vortex", "Nebula", "Quasar", "Pulsar", "Comet", "Meteor", "Cosmos", "Astral", "Lunar",
    "Twilight", "Dusk", "Gloom", "Shadow", "Umbra", "Void", "Rift", "Chasm", "Echo", "Tremor",
    "Seismic", "Fracture", "Shatter", "Crystal", "Prism", "Spectrum", "Aura", "Oasis", "Dune",
    "Indigo", "Violet", "Crimson", "Scarlet", "Amber", "Topaz", "Saffron", "Velour", "Satin",
    "Suede", "Marble", "Ivory", "Ashen", "Cinder", "Scorch", "Magma", "Phoenix", "Seraph",
    "Oracle", "Sphinx", "Mystic", "Rune", "Sigil", "Talisman", "Relic", "Idol", "Totem", "Halo",
    "Drift", "Haze", "Vapor", "Frost", "Glacier", "Tundra", "Abyss", "Trench", "Fathom",
    "Riptide", "Tempest", "Cyclone", "Static", "Pylon", "Reactor", "Plasma", "Quartz", "Garnet",
    "Jade", "Opal", "Slate", "Basalt", "Granite", "Flint", "Smolder", "Soot", "Coal", "Pitch",
    "Jet", "Ink", "Noir", "Mojave", "Polar", "Arctic", "Borealis", "Zenith", "Apex", "Vertex",
    "Crest", "Summit", "Ridge", "Canyon", "Gorge", "Delta", "Lagoon", "Reef", "Tide", "Current",
    "Undertow", "Maelstrom", "Squall", "Monsoon", "Typhoon", "Gale", "Mistral", "Sirocco",
    "Cobalt2", "Onyxx",
    # --- expansion (2026-06-29): catalog grew past the original ~140; more on-brand single
    # words (dark / cosmic / luxe / mystic / mythic / gem / terrain / weather) ---
    "Cinnabar", "Vermilion", "Cerulean", "Verdigris", "Malachite", "Lapis", "Azurite", "Pyrite",
    "Hematite", "Obelisk", "Monolith", "Cairn", "Menhir", "Spire", "Bastion", "Citadel",
    "Rampart", "Turret", "Vault", "Crypt", "Catacomb", "Sepulcher", "Ossuary", "Reliquary",
    "Censer", "Thurible", "Chalice", "Goblet", "Diadem", "Coronet", "Scepter", "Orb", "Regalia",
    "Brocade", "Damask", "Taffeta", "Chiffon", "Organza", "Cashmere", "Mohair", "Angora",
    "Vicuna", "Pashmina", "Lacquer", "Veneer", "Patina", "Gilt", "Filigree", "Inlay", "Mosaic",
    "Fresco", "Tessera", "Porphyry", "Travertine", "Alabaster", "Soapstone", "Serpentine",
    "Gneiss", "Schist", "Quartzite", "Dolomite", "Calcite", "Feldspar", "Mica", "Olivine",
    "Peridot", "Zircon", "Spinel", "Beryl", "Tourmaline", "Citrine", "Morganite", "Tanzanite",
    "Alexandrite", "Aquamarine", "Sunstone", "Moonstone", "Bloodstone", "Tigereye", "Labradorite",
    "Carnelian", "Chalcedony", "Sardonyx", "Chrysoprase", "Heliotrope", "Aventurine", "Eventide",
    "Gloaming", "Nightfall", "Daybreak", "Aurora", "Corona", "Penumbra", "Solstice", "Equinox",
    "Perigee", "Apogee", "Syzygy", "Ecliptic", "Meridian", "Azimuth", "Parallax", "Nadir",
    "Empyrean", "Firmament", "Ether", "Aether", "Ion", "Photon", "Graviton", "Neutrino",
    "Tachyon", "Boson", "Lepton", "Hadron", "Fermion", "Singularity", "Wormhole", "Magnetar",
    "Supernova", "Nova", "Ember", "Pyre", "Inferno", "Brimstone", "Sulfur", "Tungsten",
    "Titanium", "Chromium", "Vanadium", "Iridium", "Osmium", "Rhodium", "Palladium", "Platinum",
    "Mercury", "Antimony", "Bismuth", "Selenium", "Tellurium", "Gallium", "Indium", "Thallium",
    "Specter", "Revenant", "Wendigo", "Lich", "Ghoul", "Nightmare", "Incubus", "Gorgon",
    "Basilisk", "Chimera", "Manticore", "Wyvern", "Hydra", "Kraken", "Leviathan", "Behemoth",
    "Golem", "Valkyrie", "Naga", "Djinn", "Ifrit", "Marid", "Efreet", "Cathedral", "Sanctum",
    "Nave", "Apse", "Transept", "Cloister", "Belfry", "Steeple", "Minaret", "Ziggurat", "Pagoda",
    "Pantheon", "Acropolis", "Necropolis", "Colossus", "Whirlpool", "Eddy", "Surge", "Swell",
    "Breaker", "Whitecap", "Brine", "Flurry", "Blizzard", "Whiteout", "Avalanche", "Cornice",
    "Sastrugi", "Permafrost", "Moraine", "Crevasse", "Serac", "Icefall", "Floe", "Berg",
]


def assign_unique(loops):
    """loops: list of {id, title, key, bpm, rating}.
    Returns (keep, changes): keep = {id: kept_word}; changes = {id: new_word}."""
    groups = defaultdict(list)
    for L in loops:
        groups[L["title"]].append(L)

    used = set()
    keep = {}
    for title, ms in groups.items():
        best = sorted(ms, key=lambda L: (-(L.get("rating") or 0), str(L["id"])))[0]
        keep[best["id"]] = title
        used.add(title)

    avail = [w for w in POOL if w not in used]
    changes = {}
    i = 0
    for L in sorted(loops, key=lambda L: str(L["id"])):
        if L["id"] in keep:
            continue
        if i >= len(avail):
            raise RuntimeError("name pool exhausted — add more words to POOL")
        new = avail[i]; i += 1
        changes[L["id"]] = new
        used.add(new)
    return keep, changes
