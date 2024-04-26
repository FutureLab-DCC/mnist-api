#!/bin/bash

# Start minikube

mkbe_test=`minikube status | grep host | cut -d : -f 2`

if [ $mkbe_test = "Stopped" ]; then
  minikube start --cpus=10 --memory='20g' --disk-size='40g'
fi


# MongoDB deployment

mongo_test=`helm status my-mongodb | grep VERSION`
if [ -z "$mongo_test" ]; then
  helm install my-mongodb  --set auth.rootPassword=futurelab,auth.username=futurelab,auth.password=futurelab,auth.database=admin bitnami/mongodb --version 14.13.0

fi

kubectl port-forward --namespace default svc/my-mongodb 27017:27017 &

# DFS deployment

dfs_test=`kubectl get all | grep service/nfs-server-provisioner`
if [ -z "$dfs_test" ]; then
  helm install nfs-server nfs-ganesha-server-and-external-provisioner/nfs-server-provisioner --set=persistence.accessMode="ReadWriteMany"
fi

export NFS_CLUSTER_IP=$(kubectl get svc/nfs-server-nfs-server-provisioner -o jsonpath='{.spec.clusterIP}')

kubectl get all

