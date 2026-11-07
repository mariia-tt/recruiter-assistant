from recruiter.data.clean import clean_text, fix_glued_words, mask_pii


def test_mask_pii():
    out = mask_pii("Mail john.doe@gmail.com or call (555) 123-4567, see www.site.com")
    assert "[EMAIL]" in out and "[PHONE]" in out and "[URL]" in out
    assert "gmail" not in out and "4567" not in out


def test_fix_glued_words_splits_sections_and_months():
    assert fix_glued_words("HighlightsMicrosoft Office") == "Highlights Microsoft Office"
    assert fix_glued_words("graduated inMay 2010") == "graduated in May 2010"
    assert fix_glued_words("data analysisEducation") == "data analysis Education"


def test_fix_glued_words_keeps_product_names():
    assert fix_glued_words("PowerPoint and JavaScript") == "PowerPoint and JavaScript"


def test_clean_text_whitespace():
    assert clean_text("  a   b \n\n\n\n c ") == "a b\n\nc"
