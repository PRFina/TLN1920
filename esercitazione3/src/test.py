import text_summarization as summ
import topic_extraction as te
import data_manager as dm

from pathlib import Path

if __name__ == '__main__':
    nasari = dm.Nasari(Path('data/dd-small-nasari-15.txt'))
    doc = dm.parse_document_sentence(Path('data/text-documents/Life-indoors.txt'))

    summarizer = summ.TextSummarizer(main_extractor=te.TitleExtractor(nasari),
                                     chunk_extractor=te.TopicExtractor(nasari))
    s = summarizer.get_summary(doc, 90, debug=False)
    output_path = Path('output')
    output_path.mkdir(exist_ok=True) 

    with (output_path / 'summary.txt').open('w') as file:
        file.write(s)
    print(s)
