"""
@title

@description

"""
import argparse
from pathlib import Path

from pdfminer.high_level import extract_text
from pypdf import PdfReader

from playtime import project_properties

PDF_DIR = Path(project_properties.data_dir, 'pdfs')
PDF_PATHS = [each_path for each_path in PDF_DIR.glob('*.pdf')]


def main(main_args):
    for each_path in PDF_PATHS:
        print(f'{each_path}')
        text = extract_text(each_path)
        print(text)
        print('-' * 80)

        reader = PdfReader(each_path)
        for page in reader.pages:
            text = page.extract_text()
            print(text)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
