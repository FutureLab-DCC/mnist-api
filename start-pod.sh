#!/bin/bash



check=`kubectl get all | grep pod/mnist-experiment-pod`
if [ -z "$check" ]; then
  cd ./pod
  cat nfs-pv-pvc.yaml | sed "s/{{NFS_CLUSTER_IP}}/$NFS_CLUSTER_IP/g" | kubectl apply -f -
  kubectl apply -f mnist-pod.yaml
fi


kubectl get pvc
kubectl get all

