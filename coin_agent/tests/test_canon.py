from agent import canon, schema


def test_synonym_normalization():
    assert canon.normalize("navy blue", "COLOR") == "navy"
    assert canon.normalize("Off-White", "COLOR") == "off_white"
    assert canon.normalize("butcher block", "MATERIAL") == "butcher_block"
    assert canon.normalize("not a real color", "COLOR") is None


def test_relation_examples_from_spec():
    # spec §3.3: "NAVY/WHITE is FAR, NAVY/BLUE is NEAR"
    assert canon.relation("COLOR", "navy", "white") == canon.FAR
    assert canon.relation("COLOR", "navy", "blue") == canon.NEAR


def test_relation_same_is_identity():
    for vocab_name, values in schema.VOCAB.items():
        for v in values:
            assert canon.relation(vocab_name, v, v) == canon.SAME


def test_relation_is_symmetric_and_total():
    for vocab_name, values in schema.VOCAB.items():
        for a in values:
            for b in values:
                rel_ab = canon.relation(vocab_name, a, b)
                rel_ba = canon.relation(vocab_name, b, a)
                assert rel_ab == rel_ba, f"{vocab_name}: relation({a},{b})={rel_ab} != relation({b},{a})={rel_ba}"
                assert rel_ab in (canon.SAME, canon.NEAR, canon.FAR)


def test_far_pairs_are_disjoint_from_near_clusters():
    """A pair can't be listed as both NEAR (same cluster) and FAR — that would be a
    self-contradictory table entry.
    """
    for vocab_name in schema.VOCAB:
        far_pairs = {frozenset(p) for p in canon._FAR_PAIRS.get(vocab_name, [])}
        near_clusters = canon._NEAR_CLUSTERS.get(vocab_name, [])
        for cluster in near_clusters:
            for a in cluster:
                for b in cluster:
                    if a != b:
                        assert frozenset((a, b)) not in far_pairs, (
                            f"{vocab_name}: {a}/{b} is in both a NEAR cluster and the FAR table"
                        )


def test_find_in_text_prefers_longest_match():
    assert canon.find_in_text("The cabinet is dark blue.", "COLOR") == "dark_blue"
    assert canon.find_in_text("It looks tan to me.", "COLOR") == "tan"
    assert canon.find_in_text("no color mentioned here", "COLOR") is None


def test_find_in_text_word_boundary():
    # "tan" must not match inside "stand"
    assert canon.find_in_text("The lamp stands in the corner.", "COLOR") is None
