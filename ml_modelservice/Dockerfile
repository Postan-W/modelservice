FROM harbor.stress.com:8443/library/ml-model-service:base
COPY machine_learning ./machine_learning
COPY jdk1.8.0_201 /root/jdk1.8.0_201/
ENV JAVA_HOME /root/jdk1.8.0_201
ENV PATH $JAVA_HOME/bin:$PATH
ENV PATH $JAVA_HOME/jre/bin:$PATH
RUN chmod -R a+x /root/jdk1.8.0_201/
COPY run.sh /run.sh
COPY xgboost4j-0.90.jar /usr/local/lib/python3.7/site-packages/pyspark/jars/
COPY xgboost4j-spark-0.90.jar /usr/local/lib/python3.7/site-packages/pyspark/jars/
RUN  apt-get install libgomp1
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ sklearn
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ sklearn2pmml
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ pypmml
RUN apt-get update && apt-get -y install curl
ENTRYPOINT ["/bin/bash","/run.sh"]

