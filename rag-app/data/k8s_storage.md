# Kubernetes Storage

A PersistentVolume (PV) is a piece of storage in the cluster provisioned by an
administrator or dynamically using Storage Classes. A PersistentVolumeClaim
(PVC) is a request for storage by a user.

The access modes for a PersistentVolume are ReadWriteOnce, ReadOnlyMany,
ReadWriteMany, and ReadWriteOncePod. ReadWriteOnce allows the volume to be
mounted as read-write by a single node.

A StorageClass provides a way for administrators to describe the classes of
storage they offer. The reclaim policy of a PersistentVolume can be Retain,
Delete, or Recycle. The default reclaim policy for dynamically provisioned
volumes is Delete.
