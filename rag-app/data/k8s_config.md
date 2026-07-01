# ConfigMaps and Secrets

A ConfigMap is an API object used to store non-confidential configuration data
in key-value pairs. Pods can consume ConfigMaps as environment variables,
command-line arguments, or as configuration files in a volume.

A Secret is similar to a ConfigMap but is intended to hold confidential data
such as passwords, OAuth tokens, and SSH keys. By default, Secrets are stored
unencrypted as base64-encoded strings in etcd. Base64 encoding is not encryption.

To encrypt Secrets at rest, you must enable encryption at the etcd level using an
EncryptionConfiguration. The maximum size of a Secret is 1 MiB.
