How to Use Private Data Source
==============================

To enable users to fetch custom data from cloud we expose basic tooling at their disposal. In this document you will
learn how to fetch the data at bulk or per file.

.. important::

    Whenever you use a CLI from the **Noether** framework - your data stays on your local machine!

    Emmi AI doesn't store or collect your data.


Interface and Examples
----------------------

``noether.io`` CLI supports two types of executions:

1. ``noether-data <SERVICE_NAME> <COMMAND>`` — specify a service name (each has unique commands).
2. ``noether-data <COMMAND>`` — run a global command.

Hugging Face
~~~~~~~~~~~~

.. code-block:: bash

    noether-data huggingface estimate EmmiAI/AB-UPT
    noether-data huggingface ext EmmiAI/NeuralDEM .th ~/data --type model --manifest-out manifest.json

The ``ext`` command downloads all ``.th`` files from ``EmmiAI/NeuralDEM`` into ``~/data``.
The ``--manifest-out`` option writes a manifest for integrity checks.

AWS
~~~

.. code-block:: bash

    noether-data aws estimate noaa-goes16 ABI-L1b-RadC/2023/001/00/
    noether-data aws fetch my-bucket data/prefix/ ./data --extension .parquet --manifest-out s3-manifest.json

The ``fetch`` command downloads only ``.parquet`` files into ``./data``, while creating a manifest file.

Verification
------------

Verification determines whether files are complete. If ``manifest.json`` exists, corrupted or missing files can be
redownloaded:

.. code-block:: bash

    noether-data verification check -r ./data -m manifest.json --action redownload

If no manifest exists, create one with:

.. code-block:: bash

    noether-data verification build -r ./data -m manifest.json

To explore all options, use the ``--help`` flag.

Other Links
-----------
:doc:`Working with the CLI <../working_with_cli>`

