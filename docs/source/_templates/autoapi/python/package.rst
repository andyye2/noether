{{ obj.name }}
{{ "=" * obj.name|length }}

{# Optional: one-line summary #}
{% if obj.docstring and obj.docstring|striptags|trim %}
{{ obj.docstring|striptags|trim }}
{% endif %}

.. toctree::
   :maxdepth: 1

{# Subpackages first #}
{% for subpackage in obj.subpackages %}
   {{ subpackage.relative_path }}
{% endfor %}

{# Modules in this package #}
{% for module in obj.modules %}
   {{ module.relative_path }}
{% endfor %}
