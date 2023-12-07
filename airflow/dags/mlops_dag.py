import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
import pendulum
import datetime as dt

args = {
    'owner': 'admin',
    'start_date': dt.datetime(2023, 12, 7),
    'retries': 1,
    'retry_delays': dt.timedelta(minutes=1),
    'depends_on_past': False,
    'provide_context': True
}

with DAG(
    dag_id='mlops_dag',
    default_args=args,
    schedule_interval="@hourly",
    tags=['mlops', 'score'],
) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="python3 /home/ml_srv_admin/PycharmProjects/MLops_Flow/scripts/get_data.py",
                            dag=dag)
    data_preprocessing = BashOperator(task_id='data_preprocessing',
                            bash_command="python3 /home/ml_srv_admin/PycharmProjects/MLops_Flow/scripts/data_preprocessing.py",
                            dag=dag)
    model_fit = BashOperator(task_id='model_fit',
                            bash_command="python3 /home/ml_srv_admin/PycharmProjects/MLops_Flow/scripts/model_fit.py",
                            dag=dag)
    model_test = BashOperator(task_id='model_test',
                            bash_command="python3 /home/ml_srv_admin/PycharmProjects/MLops_Flow/scripts/model_test.py",
                            dag=dag)
    get_data >> data_preprocessing >> model_fit >> model_test
