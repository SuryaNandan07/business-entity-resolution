import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.scalability_report import audit

class AuditTests(unittest.TestCase):
    def test_features_and_ids_are_both_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('reference','current'):
                db=sqlite3.connect(root/f'{name}.sqlite')
                db.execute('CREATE TABLE completed(s1 TEXT)');db.execute("INSERT INTO completed VALUES ('q')")
                db.execute('CREATE TABLE pairs(s1 TEXT,candidate TEXT,features BLOB)')
                db.execute('INSERT INTO pairs VALUES (?,?,?)',('q','S2-a',b'features'))
                db.commit();db.close()
            with patch('src.scalability_report.OUT',root):
                result=audit(root/'reference.sqlite',root/'current.sqlite')
                self.assertEqual(result['compared_queries'],1)
                db=sqlite3.connect(root/'current.sqlite');db.execute("UPDATE pairs SET features=?",(b'changed',));db.commit();db.close()
                with self.assertRaises(ValueError):audit(root/'reference.sqlite',root/'current.sqlite')
                db=sqlite3.connect(root/'current.sqlite');db.execute("UPDATE pairs SET candidate='S2-b'");db.commit();db.close()
                with self.assertRaises(ValueError):audit(root/'reference.sqlite',root/'current.sqlite')

if __name__=='__main__':unittest.main()
