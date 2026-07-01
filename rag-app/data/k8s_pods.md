# Kubernetes Pods

A Pod is the smallest deployable unit in Kubernetes. A Pod encapsulates one or
more containers that share the same network namespace and storage volumes.
Containers within a Pod always run on the same node and are scheduled together.

Pods are considered ephemeral. When a Pod is deleted, its data in the container
filesystem is lost unless a persistent volume is attached. Each Pod receives its
own unique IP address within the cluster.

A Pod can be in one of five phases: Pending, Running, Succeeded, Failed, or
Unknown. The Pending phase means the Pod has been accepted but one or more
containers have not yet been created. The default restart policy for a Pod is
Always.
