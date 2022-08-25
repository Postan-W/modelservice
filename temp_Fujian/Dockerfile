FROM anomalydetection:base
COPY service /root/service/
COPY run.sh /run.sh
RUN pip install flask
RUN pip install pandas==1.1.5
RUN pip install hdfs
ENTRYPOINT ["/bin/bash","/run.sh"]