import pandas as pd

from scripts.create_siddata_dataset import get_data

def get_classes(data_base, what):
    assert what in ["Fachbereich"]
    name_number = get_data().set_index("Name")["VeranstaltungsNummer"].to_dict()
    fachbereich_per_course = {k: int(v.split(".", 1)[0]) for k, v in name_number.items() if
                              v.split(".", 1)[0].isdigit() and int(v.split(".", 1)[0]) <= 10}  # There are 10 FBs
    fb_mapper = {1: "Sozial,Kultur,Kunst", 3: "Theologie,Lehramt,Musik", 4: "Physik", 5: "Bio,Chemie", 6: "Mathe,Info",
                 7: "Sprache,Literatur", 8: "Humanwiss", 9: "Wiwi", 10: "Rechtswiss"}
    fachbereiche = [fachbereich_per_course.get(name, 0) for name in name_number]
    return dict(zip(name_number.keys(), [fb_mapper.get(i, "Unknown") for i in fachbereiche]))


if __name__ == "__main__":
    from src.static.settings import DATA_BASE
    tmp = get_classes(DATA_BASE, "Fachbereich")
    print(tmp)