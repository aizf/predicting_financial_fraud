# source venv/bin/activate
# gunicorn -c gunicorn_conf.py app:app
# pstree -ap|grep gunicorn
# kill -9 pid
import multiprocessing
# bind = '127.0.0.1:5000'
bind = '0.0.0.0:5000'
chdir = '/home/ubuntu/GitHub/predicting_financial_fraud'
workers = multiprocessing.cpu_count() * 2 + 1    #进程数
threads = 2 #指定每个进程开启的线程数
pidfile = 'log/gunicorn.pid'
accesslog = "log/gunicorn_access.log"      #访问日志文件
errorlog = "log/gunicorn_error.log"        #错误日志文件