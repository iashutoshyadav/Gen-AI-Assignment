# Kubernetes Deployments

A Deployment provides declarative updates for Pods and ReplicaSets. You describe
a desired state in a Deployment, and the Deployment controller changes the actual
state to the desired state at a controlled rate.

The default deployment strategy is RollingUpdate, which replaces Pods gradually
to ensure zero downtime. The other strategy is Recreate, which terminates all
existing Pods before creating new ones.

Deployments support rollback. If a rollout fails, you can revert to a previous
revision using the rollout undo command. By default, Kubernetes retains the last
ten revisions in the revision history.
