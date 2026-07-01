# Kubernetes Services

A Service is an abstraction that defines a logical set of Pods and a policy by
which to access them. Because Pods are ephemeral and their IPs change, Services
provide a stable endpoint for a group of Pods selected by labels.

There are four main Service types. ClusterIP exposes the Service on an internal
cluster IP and is the default type. NodePort exposes the Service on each node's
IP at a static port. LoadBalancer exposes the Service externally using a cloud
provider's load balancer. ExternalName maps the Service to a DNS name.

kube-proxy runs on each node and is responsible for implementing the Service
abstraction by managing network rules. Services use label selectors to determine
which Pods receive traffic.
