from pathlib import Path

from conftest import load_script


def test_is_compiled_mp4():
    ef = load_script("extract_frames.py", ["is_compiled_mp4"])
    assert ef.is_compiled_mp4(Path("SIGHTING_T8_3.mp4"))
    assert ef.is_compiled_mp4(Path("sighting_t8_3.MP4"))
    assert not ef.is_compiled_mp4(Path("IMAG0014.MP4"))
    assert not ef.is_compiled_mp4(Path("SIGHTING_T8_3.jpg"))


def test_review_app_serves_ui_and_frames(tmp_path):
    ra = load_script("review_app.py", ["app", "build_index"])
    frames = tmp_path / "Frames"
    (frames / "S1").mkdir(parents=True)
    (frames / "S1" / "a.jpg").write_bytes(b"jpg")
    ra.FRAMES_DIR = frames
    ra.REVIEWS_FILE = tmp_path / "reviews.csv"
    ra.build_index()

    client = ra.app.test_client()
    assert client.get("/").status_code == 200
    data = client.get("/api/frames").get_json()
    assert data["total"] == 1 and data["frames"][0]["id"] == "S1/a.jpg"
    assert client.get("/img/S1/a.jpg").data == b"jpg"

    r = client.post("/api/review", json={"id": "S1/a.jpg", "presence": "present", "quality": "good"})
    assert r.get_json()["done"] == 1
    assert "S1/a.jpg,S1,a.jpg,present,good" in ra.REVIEWS_FILE.read_text()


def test_yolo_label_round_trip():
    # Converter lives in scripts/prepare_data.py (skipped until it exists).
    pd = load_script("prepare_data.py", ["yolo_lines_to_entries", "entries_to_yolo_lines"])
    lines = ["0 0.5 0.25 0.1 0.2", "", "0 0.123456 0.654321 0.3 0.4"]
    entries = pd.yolo_lines_to_entries(lines)
    assert [e["instance_id"] for e in entries] == [0, 1]
    assert entries[0]["color_rgb"] != entries[1]["color_rgb"]
    assert pd.entries_to_yolo_lines(entries) == [
        "0 0.500000 0.250000 0.100000 0.200000",
        "0 0.123456 0.654321 0.300000 0.400000",
    ]
    assert pd.entries_to_yolo_lines([]) == []
