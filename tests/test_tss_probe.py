import pytest

from tools.tss_probe import (
    find_latest_teacher_checkpoint,
    parse_sample_list_line,
    select_probe_records,
    split_wsi_ids_by_tss,
)


def test_parse_sample_list_line_extracts_tss_and_coordinates():
    record = parse_sample_list_line("/block/TCGA/TCGA-BH-A0BM-01Z.svs 123 456 0")

    assert record.path == "/block/TCGA/TCGA-BH-A0BM-01Z.svs"
    assert record.wsi_id == "TCGA-BH-A0BM-01Z.svs"
    assert record.tss == "BH"
    assert record.x == 123
    assert record.y == 456
    assert record.level == 0


def test_parse_sample_list_line_rejects_non_tcga_names():
    with pytest.raises(ValueError, match="Cannot extract TSS"):
        parse_sample_list_line("/block/TCGA/not_tcga.svs 0 0 0")


def test_split_wsi_ids_by_tss_holds_out_wsis_not_tiles():
    wsi_by_tss = {
        "AA": {f"AA_{i}" for i in range(8)},
        "BB": {f"BB_{i}" for i in range(8)},
    }

    split = split_wsi_ids_by_tss(
        wsi_by_tss,
        max_wsis_per_class=6,
        test_size=0.33,
        seed=7,
    )

    assert set(split.train_wsi_ids).isdisjoint(split.test_wsi_ids)
    assert len(split.class_names) == 2
    assert all(label in split.train_labels for label in ("AA", "BB"))
    assert all(label in split.test_labels for label in ("AA", "BB"))


def test_split_wsi_ids_by_tss_keeps_rare_classes_with_two_wsis():
    wsi_by_tss = {
        "AA": {"AA_train_or_test_0", "AA_train_or_test_1"},
        "BB": {"BB_train_or_test_0", "BB_train_or_test_1"},
    }

    split = split_wsi_ids_by_tss(
        wsi_by_tss,
        max_wsis_per_class=10,
        test_size=0.3,
        seed=3,
    )

    assert len(split.train_wsi_ids) == 2
    assert len(split.test_wsi_ids) == 2
    assert sorted(split.train_labels) == ["AA", "BB"]
    assert sorted(split.test_labels) == ["AA", "BB"]


def test_select_probe_records_limits_tiles_per_wsi_and_uses_split_labels():
    records_by_wsi = {
        "AA_0": [parse_sample_list_line(f"/data/TCGA-AA-000.svs {i} 0 0") for i in range(5)],
        "AA_1": [parse_sample_list_line(f"/data/TCGA-AA-001.svs {i} 0 0") for i in range(5)],
        "BB_0": [parse_sample_list_line(f"/data/TCGA-BB-000.svs {i} 0 0") for i in range(5)],
        "BB_1": [parse_sample_list_line(f"/data/TCGA-BB-001.svs {i} 0 0") for i in range(5)],
    }

    selected = select_probe_records(
        records_by_wsi,
        wsi_ids=["AA_0", "BB_1"],
        max_tiles_per_wsi=2,
        seed=11,
    )

    assert len(selected.records) == 4
    assert selected.labels == ["AA", "AA", "BB", "BB"]
    assert selected.wsi_ids == ["AA_0", "AA_0", "BB_1", "BB_1"]


def test_find_latest_teacher_checkpoint_uses_highest_training_step(tmp_path):
    older = tmp_path / "eval" / "training_2500"
    newer = tmp_path / "eval" / "training_10000"
    manual = tmp_path / "eval" / "manual_99999"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    manual.mkdir(parents=True)
    (older / "teacher_checkpoint.pth").write_text("older")
    (newer / "teacher_checkpoint.pth").write_text("newer")
    (manual / "teacher_checkpoint.pth").write_text("manual")

    assert find_latest_teacher_checkpoint(str(tmp_path)) == str(newer / "teacher_checkpoint.pth")
