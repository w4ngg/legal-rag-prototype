from legal_ir_prototype.evaluation.legal_references import (
    compare_references, extract_references, source_references,
)


def test_numbered_statute_clause_point_and_normalized_authority():
    refs = extract_references("Theo điểm a khoản 2 Điều 5 Nghị định số 012/2022/NĐ-CP.")
    assert len(refs) == 1
    assert (refs[0].document, refs[0].article, refs[0].clause, refs[0].point) == ("12/2022/ND-CP", "5", "2", "a")
    ascii_refs = extract_references("diem a khoan 2 Dieu 5 Nghi dinh 12/2022/ND-CP")
    assert compare_references(refs, ascii_refs)["anchor_coverage"] == 1


def test_same_article_in_different_statutes_is_not_a_match():
    target = extract_references("Khoản 1 Điều 5 Nghị định 12/2022/NĐ-CP")
    other = extract_references("Khoản 1 Điều 5 Nghị định 13/2022/NĐ-CP")
    assert compare_references(target, other)["anchor_coverage"] == 0
    missing_clause = extract_references("Điều 5 Nghị định 12/2022/NĐ-CP")
    result = compare_references(target, missing_clause)
    assert result["document_coverage"] == 1
    assert result["anchor_coverage"] == 0


def test_multiple_statutes_and_multiple_articles_keep_their_scope():
    refs = extract_references("Khoản 1 Điều 5 Nghị định 12/2022/NĐ-CP; khoản 2 Điều 5 Nghị định 13/2022/NĐ-CP.")
    assert {(r.document, r.article, r.clause) for r in refs} == {
        ("12/2022/ND-CP", "5", "1"), ("13/2022/ND-CP", "5", "2")}
    refs = extract_references("Điều 5 và Điều 6 Nghị định 12/2022/NĐ-CP.")
    assert {(r.document, r.article) for r in refs} == {("12/2022/ND-CP", "5"), ("12/2022/ND-CP", "6")}
    refs = extract_references("Điều 5 Nghị định 12/2022/NĐ-CP và Điều 6 Nghị định 13/2022/NĐ-CP")
    assert {(r.document, r.article) for r in refs} == {("12/2022/ND-CP", "5"), ("13/2022/ND-CP", "6")}


def test_unresolved_article_does_not_borrow_a_distant_statute():
    refs = extract_references("Nghị định 12/2022/NĐ-CP.\nĐiều 5 về trách nhiệm.")
    assert any(r.article == "5" and r.document is None for r in refs)
    unresolved = compare_references(extract_references("Điều 5"), refs)
    assert unresolved["anchor_coverage"] is None
    assert unresolved["unresolved_reference_anchor_count"] == 1


def test_named_law_and_source_metadata_resolve_to_same_identity():
    target = extract_references("Theo Bộ luật Lao động năm 2019, Điều 113 khoản 1.")
    source = source_references({"text": "Điều 113 khoản 1. Nội dung văn bản.",
                                "metadata": {"document_name": "Bo-luat-lao-dong-2019-333670"}})
    assert source[0].document_inferred_from_metadata is True
    assert compare_references(target, source)["anchor_coverage"] == 1
    changed_year = extract_references("Theo Bộ luật Lao động 2012, Điều 113 khoản 1.")
    assert compare_references(changed_year, source)["anchor_coverage"] == 0


def test_numbered_metadata_can_show_document_hit_but_missing_article():
    target = extract_references("Điều 31 Thông tư 25/2022/TT-NHNN")
    source = source_references({"text": "Điều 5. Quy định về hạch toán tiền.",
                                "metadata": {"document_name": "Thong-tu-25-2022-TT-NHNN-che-do-ke-toan"}})
    result = compare_references(target, source)
    assert result["document_coverage"] == 1
    assert result["anchor_coverage"] == 0
