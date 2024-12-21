from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
sys.path.insert(0, '/opt/airflow/streaming')
from main import run_pipeline

dag = DAG(
    'aerostream_pipeline',
    default_args={'owner': 'aerostream', 'start_date': datetime(2024,1,1), 'retries': 1, 'retry_delay': timedelta(minutes=1)},
    schedule_interval='* * * * *',
    catchup=False
)

PythonOperator(task_id='run_pipeline', python_callable=run_pipeline, dag=dag)
