docker build --platform linux/amd64 -t alekseykorshuk/opt-inference:v1  .
docker push alekseykorshuk/opt-inference:v1
kubectl delete inferenceservice opt-inference
kubectl apply -f opt-inference.yaml
kubectl describe inferenceservice opt-inference
kubectl get pods