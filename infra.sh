#!/bin/bash

# Start minikube

mkbe_test=`minikube status | grep host | cut -d : -f 2`

if [ $mkbe_test = "Stopped" ]; then
  minikube start --cpus=5 --memory='4g' --disk-size='20g'
fi


# MongoDB deployment

mongo_test=`helm status my-mongodb | grep VERSION`
if [ -z "$mongo_test" ]; then
  helm install my-mongodb  --set auth.rootPassword=futurelab,auth.username=futurelab,auth.password=futurelab,auth.database=admin bitnami/mongodb --version 14.13.0

fi

kubectl port-forward --namespace default svc/my-mongodb 27017:27017 &

# DFS deployment

dfs_test=`kubectl get all | grep service/nfs-server`
if [ -z "$dfs_test" ]; then
  cd ./infra
  kubectl apply -k .
fi

kubectl get all
