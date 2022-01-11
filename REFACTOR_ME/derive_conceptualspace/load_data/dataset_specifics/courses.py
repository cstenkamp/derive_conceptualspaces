from derive_conceptualspace.cli.run_pipeline import get_data
from src.static.settings import SID_DATA_BASE

def get_classes(data_base, what, from_csv_path=SID_DATA_BASE, from_csv_name="kurse-beschreibungen.csv"):
    assert what in ["Fachbereich"]
    name_number = get_data(from_csv_path, from_csv_name).set_index("Name")["VeranstaltungsNummer"].to_dict()
    fachbereich_per_course = {k: int(v.split(".", 1)[0]) for k, v in name_number.items() if
                              v.split(".", 1)[0].isdigit() and int(v.split(".", 1)[0]) <= 10}  # There are 10 FBs
    #TODO instead use `make_classifier_dict(df.set_index("Name")["VeranstaltungsNummer"])`
    fb_mapper = {1: "Sozial,Kultur,Kunst", 3: "Theologie,Lehramt,Musik", 4: "Physik", 5: "Bio,Chemie", 6: "Mathe,Info",
                 7: "Sprache,Literatur", 8: "Humanwiss", 9: "Wiwi", 10: "Rechtswiss"}
    fachbereiche = [fachbereich_per_course.get(name, 0) for name in name_number]
    return dict(zip(name_number.keys(), [fb_mapper.get(i, "Unknown") for i in fachbereiche]))


if __name__ == "__main__":
    from src.static.settings import SPACES_DATA_BASE

    tmp = get_classes(SPACES_DATA_BASE, "Fachbereich")
    print(tmp)