import os

import boto3
from dotenv import load_dotenv

load_dotenv('.env.dev')


def make_dir_n_download(s3, bucket, key, dir, filename):
    file_dir = os.path.join(dir, filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(file_dir):
        print(f"download {key} -> {file_dir}...")
        s3.download_file(Bucket=bucket, Key=key, Filename=os.path.join(dir, filename))
    else:
        print(f"file exists : {file_dir}")


def download_vectordb():
    print(f"Download chroma.sqlite3".ljust(60, '-'))
    s3 = boto3.client(
        service_name="s3",
        region_name="ap-northeast-2",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )
    if not os.path.exists('llmrec/vector_dbs/hyeonwoo'):
        os.makedirs('llmrec/vector_dbs/hyeonwoo')
    print(f"Download Hyeonwoo's vector db & files...")
    make_dir_n_download(s3=s3, bucket='pseudorec-data', key='hyeonwoo/chroma_db_content_0614/chroma.sqlite3',
                        dir='llmrec/vector_dbs/hyeonwoo/chroma_db_content_0614', filename='chroma.sqlite3')
    make_dir_n_download(s3=s3, bucket='pseudorec-data', key='hyeonwoo/chroma_db_title_0614/chroma.sqlite3',
                        dir='llmrec/vector_dbs/hyeonwoo/chroma_db_content_0614', filename='chroma.sqlite3')
    make_dir_n_download(s3=s3, bucket='pseudorec-data', key='hyeonwoo/dictionary/actor_rec.json',
                        dir='llmrec/vector_dbs/hyeonwoo/dictionary', filename='actor_rec.json')
    make_dir_n_download(s3=s3, bucket='pseudorec-data', key='hyeonwoo/dictionary/director_rec.json',
                        dir='llmrec/vector_dbs/hyeonwoo/dictionary', filename='director_rec.json')
    make_dir_n_download(s3=s3, bucket='pseudorec-data', key='hyeonwoo/dictionary/title_rec.json',
                        dir='llmrec/vector_dbs/hyeonwoo/dictionary', filename='title_rec.json')
    make_dir_n_download(s3=s3, bucket='pseudorec-data', key='hyeonwoo/dictionary/title_synopsis_dict.json',
                        dir='llmrec/vector_dbs/hyeonwoo/dictionary', filename='title_synopsis_dict.json')

    if not os.path.exists('llmrec/vector_dbs/gyungah'):
        os.makedirs('llmrec/vector_dbs/gyungah')
    print(f"Download Gyungah's vector db & files...")
    make_dir_n_download(s3=s3, bucket='pseudorec-data', key='gyungah/chroma.sqlite3',
                        dir='llmrec/vector_dbs/gyungah', filename='chroma.sqlite3')


if __name__ == "__main__":
    print(os.getcwd())
    download_vectordb()
