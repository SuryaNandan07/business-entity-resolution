"""Small normalization tests; no dataset loading or extra dependencies.

Run from the project root: python -m unittest discover -s tests -v
"""

import unittest

from src.normalize import normalize_address, normalize_name, normalize_text


class NormalizeTextTests(unittest.TestCase):
    def test_missing_and_empty_values(self):
        for value in (None, float("nan"), "", "  \t\n "):
            with self.subTest(value=value):
                self.assertEqual(normalize_text(value), "")

    def test_business_name_variants(self):
        for value in ("ABC Technologies Pvt. Ltd.", "abc technologies pvt ltd"):
            self.assertEqual(normalize_name(value), "abc technologies pvt ltd")

    def test_address_variants(self):
        for value in ("25, MG Road", "25 MG ROAD"):
            self.assertEqual(normalize_address(value), "25 mg road")

    def test_whitespace(self):
        self.assertEqual(normalize_text("  Surya   Medical Store  "),
                         "surya medical store")
        self.assertEqual(normalize_text("ABC\t\n\u00a0 Technologies"),
                         "abc technologies")

    def test_punctuation_separates_tokens(self):
        self.assertEqual(normalize_text("A.B.C. Technologies"),
                         "a b c technologies")
        self.assertEqual(normalize_text("  A,,,B—C/D_(E)  "), "a b c d e")
        self.assertEqual(normalize_text("..."), "")

    def test_french_accents_are_preserved(self):
        self.assertEqual(normalize_name("ÉCOLE Française SARL"),
                         "école française sarl")
        self.assertEqual(normalize_address("25, Rue de l’Église, 75001 Paris"),
                         "25 rue de l église 75001 paris")

    def test_composed_and_decomposed_unicode_match(self):
        self.assertEqual(normalize_text("Cafe\u0301"), normalize_text("Café"))
        self.assertEqual(normalize_text("E\u0301COLE"), "école")

    def test_all_suffixes_are_preserved(self):
        self.assertEqual(normalize_name("LTD LLC INC PVT PRIVATE LIMITED SAS SARL"),
                         "ltd llc inc pvt private limited sas sarl")

    def test_numbers_and_leading_zeroes_are_preserved(self):
        self.assertEqual(normalize_address("00125, 25/7 MG Road 75001"),
                         "00125 25 7 mg road 75001")
        self.assertEqual(normalize_text(25), "25")
        self.assertEqual(normalize_text(0), "0")

    def test_no_country_specific_expansion(self):
        self.assertEqual(normalize_address("St. Rd. Av. Rue"), "st rd av rue")

    def test_literal_missing_words_remain_text(self):
        self.assertEqual(normalize_text("NaN"), "nan")
        self.assertEqual(normalize_text("None"), "none")

    def test_symbols_are_preserved(self):
        self.assertEqual(normalize_text("A+B €25"), "a+b €25")

    def test_normalization_is_idempotent(self):
        for value in (None, "ABC Pvt. Ltd.", "Cafe\u0301—SARL", "25, MG Road"):
            normalized = normalize_text(value)
            self.assertEqual(normalize_text(normalized), normalized)


if __name__ == "__main__":
    unittest.main()
