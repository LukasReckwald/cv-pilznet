"""
Hier wird jede Pilzart-Klasse aus dem Kaggle-Datensatz (Ordnername)
einer von zwei Kategorien zugeordnet: "edible" (essbar) oder "poisonous"
(nicht essbar / giftig).

Einordnungsregel (safety-first, "essbar" ist die Ausnahme):
- "edible" nur für Arten, die in Standard-Pilzführern eindeutig als
  Speisepilz (gegart) gelten.
- "poisonous" für alles andere: nachweislich giftige/psychoaktive Arten,
  zäh-ungenießbare Arten (v. a. viele Porlinge/Krustenpilze), Arten mit
  unklarer/umstrittener Genießbarkeit, sowie alle Nicht-Speisepilz-Organismen
  im Datensatz (Flechten wie Cladonia/Evernia/Xanthoria, der Schleimpilz
  Lycogala epidendrum). Für die binäre Klassifikationsaufgabe ist das
  bewusst grob vereinfacht.

WICHTIG: Diese Zuordnung dient ausschließlich dem ML-Klassifikationsprojekt
und ist KEINE Sammelanleitung für die Praxis. Vor einer Verwendung in einem
Bericht/einer Abgabe bitte stichprobenartig gegen eine mykologische Quelle
prüfen (z. B. gbif.org, pilzepilze.de, oder ein aktuelles Pilzbuch).
"""

from typing import Literal

CLASS_TO_LABEL: dict[str, Literal["edible", "poisonous"]] = {
    "Agaricus augustus": "edible",
    "Agaricus xanthodermus": "poisonous",  # Gelbfärbender Egerling, giftig (Magen-Darm) - Verwechslungspilz zu A. augustus
    "Amanita amerirubescens": "poisonous",  # Gattung Amanita pauschal als "poisonous" (Verwechslungsgefahr)
    "Amanita augusta": "poisonous",
    "Amanita brunnescens": "poisonous",
    "Amanita calyptroderma": "poisonous",
    "Amanita citrina": "poisonous",
    "Amanita flavoconia": "poisonous",
    "Amanita muscaria": "poisonous",
    "Amanita pantherina": "poisonous",
    "Amanita persicina": "poisonous",
    "Amanita phalloides": "poisonous",  # Grüner Knollenblätterpilz, tödlich giftig
    "Amanita rubescens": "poisonous",
    "Amanita velosa": "poisonous",
    "Apioperdon pyriforme": "edible",
    "Armillaria borealis": "edible",
    "Armillaria mellea": "edible",
    "Armillaria tabescens": "edible",
    "Artomyces pyxidatus": "edible",
    "Bjerkandera adusta": "poisonous",
    "Bolbitius titubans": "poisonous",
    "Boletus edulis": "edible",
    "Boletus pallidus": "edible",
    "Boletus reticulatus": "edible",
    "Boletus rex-veris": "edible",
    "Calocera viscosa": "poisonous",
    "Calycina citrina": "poisonous",
    "Cantharellus californicus": "edible",
    "Cantharellus cibarius": "edible",
    "Cantharellus cinnabarinus": "edible",
    "Cerioporus squamosus": "edible",
    "Cetraria islandica": "poisonous",  # Flechte, kein Speisepilz
    "Chlorociboria aeruginascens": "poisonous",
    "Chlorophyllum brunneum": "poisonous",
    "Chlorophyllum molybdites": "poisonous",
    "Chondrostereum purpureum": "poisonous",
    "Cladonia fimbriata": "poisonous",  # Flechte
    "Cladonia rangiferina": "poisonous",  # Flechte
    "Cladonia stellaris": "poisonous",  # Flechte
    "Clitocybe nebularis": "poisonous",
    "Clitocybe nuda": "edible",
    "Coltricia perennis": "poisonous",
    "Coprinellus disseminatus": "edible",
    "Coprinellus micaceus": "edible",
    "Coprinopsis atramentaria": "poisonous",  # giftig in Kombination mit Alkohol
    "Coprinopsis lagopus": "poisonous",
    "Coprinus comatus": "edible",
    "Crucibulum laeve": "poisonous",
    "Cryptoporus volvatus": "poisonous",
    "Daedaleopsis confragosa": "poisonous",
    "Daedaleopsis tricolor": "poisonous",
    "Entoloma abortivum": "edible",
    "Evernia mesomorpha": "poisonous",  # Flechte
    "Evernia prunastri": "poisonous",  # Flechte
    "Flammulina velutipes": "edible",
    "Fomes fomentarius": "poisonous",
    "Fomitopsis betulina": "poisonous",
    "Fomitopsis mounceae": "poisonous",
    "Fomitopsis pinicola": "poisonous",
    "Galerina marginata": "poisonous",  # tödlich giftig (Amatoxine)
    "Ganoderma applanatum": "poisonous",
    "Ganoderma curtisii": "poisonous",
    "Ganoderma oregonense": "poisonous",
    "Ganoderma tsugae": "poisonous",
    "Gliophorus psittacinus": "edible",
    "Gloeophyllum sepiarium": "poisonous",
    "Graphis scripta": "poisonous",  # Flechte
    "Grifola frondosa": "edible",
    "Gymnopilus luteofolius": "poisonous",
    "Gyromitra esculenta": "poisonous",  # Frühjahrslorchel, trotz Namens giftig (Gyromitrin)
    "Gyromitra gigas": "poisonous",
    "Gyromitra infula": "poisonous",
    "Hericium coralloides": "edible",
    "Hericium erinaceus": "edible",
    "Hygrophoropsis aurantiaca": "poisonous",  # Falscher Pfifferling, unverträglich
    "Hypholoma fasciculare": "poisonous",
    "Hypholoma lateritium": "edible",
    "Hypogymnia physodes": "poisonous",  # Flechte
    "Hypomyces lactifluorum": "edible",
    "Imleria badia": "edible",
    "Inonotus obliquus": "poisonous",
    "Ischnoderma resinosum": "poisonous",
    "Kuehneromyces mutabilis": "edible",
    "Laccaria ochropurpurea": "edible",
    "Lactarius deliciosus": "edible",
    "Lactarius torminosus": "poisonous",
    "Lactarius turpis": "poisonous",
    "Laetiporus sulphureus": "edible",
    "Leccinum albostipitatum": "edible",
    "Leccinum aurantiacum": "edible",
    "Leccinum scabrum": "edible",
    "Leccinum versipelle": "edible",
    "Lepista nuda": "edible",
    "Leratiomyces ceres": "poisonous",
    "Leucoagaricus americanus": "edible",
    "Leucoagaricus leucothites": "poisonous",  # giftig, gefährliche Verwechslung mit weißen Amanita-Arten
    "Lobaria pulmonaria": "poisonous",  # Flechte
    "Lycogala epidendrum": "poisonous",  # Schleimpilz, kein Speisepilz
    "Lycoperdon perlatum": "edible",
    "Lycoperdon pyriforme": "edible",
    "Macrolepiota procera": "edible",
    "Merulius tremellosus": "poisonous",
    "Mutinus ravenelii": "poisonous",
    "Mycena haematopus": "poisonous",
    "Mycena leaiana": "poisonous",
    "Nectria cinnabarina": "poisonous",
    "Omphalotus illudens": "poisonous",  # Ölbaumtrichterling-Verwandter, giftig
    "Omphalotus olivascens": "poisonous",
    "Panaeolus papilionaceus": "poisonous",
    "Panellus stipticus": "poisonous",
    "Parmelia sulcata": "poisonous",  # Flechte
    "Paxillus involutus": "poisonous",  # Kahler Krempling, kumulativ tödlich giftig
    "Peltigera aphthosa": "poisonous",  # Flechte
    "Peltigera praetextata": "poisonous",  # Flechte
    "Phaeolus schweinitzii": "poisonous",
    "Phaeophyscia orbicularis": "poisonous",  # Flechte
    "Phallus impudicus": "poisonous",
    "Phellinus igniarius": "poisonous",
    "Phellinus tremulae": "poisonous",
    "Phlebia radiata": "poisonous",
    "Phlebia tremellosa": "poisonous",
    "Pholiota aurivella": "poisonous",
    "Pholiota squarrosa": "poisonous",
    "Phyllotopsis nidulans": "poisonous",
    "Physcia adscendens": "poisonous",  # Flechte
    "Platismatia glauca": "poisonous",  # Flechte
    "Pleurotus ostreatus": "edible",
    "Pleurotus pulmonarius": "edible",
    "Psathyrella candolleana": "poisonous",
    "Pseudevernia furfuracea": "poisonous",  # Flechte
    "Pseudohydnum gelatinosum": "edible",
    "Psilocybe azurescens": "poisonous",  # psychoaktiv/giftig
    "Psilocybe caerulescens": "poisonous",
    "Psilocybe cubensis": "poisonous",
    "Psilocybe cyanescens": "poisonous",
    "Psilocybe ovoideocystidiata": "poisonous",
    "Psilocybe pelliculosa": "poisonous",
    "Retiboletus ornatipes": "poisonous",  # extrem bitter, ungenießbar
    "Rhytisma acerinum": "poisonous",
    "Sarcomyxa serotina": "edible",
    "Sarcoscypha austriaca": "edible",
    "Sarcosoma globosum": "poisonous",
    "Schizophyllum commune": "poisonous",
    "Stereum hirsutum": "poisonous",
    "Stereum ostrea": "poisonous",
    "Stropharia aeruginosa": "poisonous",
    "Stropharia ambigua": "poisonous",
    "Suillus americanus": "edible",
    "Suillus granulatus": "edible",
    "Suillus grevillei": "edible",
    "Suillus luteus": "edible",
    "Suillus spraguei": "edible",
    "Tapinella atrotomentosa": "poisonous",
    "Trametes betulina": "poisonous",
    "Trametes gibbosa": "poisonous",
    "Trametes hirsuta": "poisonous",
    "Trametes ochracea": "poisonous",
    "Trametes versicolor": "poisonous",
    "Tremella mesenterica": "edible",
    "Trichaptum biforme": "poisonous",
    "Tricholoma murrillianum": "edible",
    "Tricholomopsis rutilans": "poisonous",
    "Tylopilus felleus": "poisonous",  # Gallenröhrling, extrem bitter/ungenießbar
    "Tylopilus rubrobrunneus": "poisonous",
    "Urnula craterium": "poisonous",
    "Verpa bohemica": "poisonous",
    "Volvopluteus gloiocephalus": "edible",
    "Vulpicida pinastri": "poisonous",  # Flechte
    "Xanthoria parietina": "poisonous",  # Flechte
}

# Beide Zielklassen für die binäre Klassifikation (fix, nicht ändern)
BINARY_CLASSES = ["edible", "poisonous"]


def get_label_for_class(class_name: str) -> str:
    """
    Gibt 'edible' oder 'poisonous' für eine gegebene Pilzart zurück.
    Wirft einen Fehler, falls die Klasse noch nicht im Mapping ist -
    das ist Absicht, damit nichts versehentlich falsch zugeordnet wird.
    """
    if class_name not in CLASS_TO_LABEL:
        raise KeyError(
            f"Klasse '{class_name}' ist noch nicht in CLASS_TO_LABEL "
            f"eingetragen. Bitte in class_mapping.py ergänzen."
        )
    return CLASS_TO_LABEL[class_name]


def validate_mapping(found_classes: list[str]) -> None:
    """
    Prüft, ob für alle im Datensatz gefundenen Ordner/Klassen ein Mapping
    existiert. Gibt eine verständliche Fehlermeldung mit den fehlenden
    Klassen aus, statt nur irgendwo mitten im Training zu crashen.
    """
    missing = [c for c in found_classes if c not in CLASS_TO_LABEL]
    if missing:
        raise ValueError(
            "Folgende Klassen fehlen noch im Mapping (class_mapping.py):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
    print(f"OK: Alle {len(found_classes)} Klassen sind im Mapping enthalten.")


if __name__ == "__main__":
    # Kleiner Selbsttest
    test_classes = list(CLASS_TO_LABEL.keys())
    validate_mapping(test_classes)
    for c in test_classes[:3]:
        print(f"{c} -> {get_label_for_class(c)}")