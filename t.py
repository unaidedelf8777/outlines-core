from outlines_core import Index, Vocabulary, Guide
from outlines_core.json_schema import build_regex_from_schema

simple_schema = """{
        "$defs": {
            "Armor": {
                "enum": ["leather", "chainmail", "plate"],
                "title": "Armor",
                "type": "string"
            }
        },
        "properties": {
            "name": {"maxLength": 10, "title": "Name", "type": "string"},
            "age": {"title": "Age", "type": "integer"},
            "armor": {"$ref": "#/$defs/Armor"},
            "strength": {"title": "Strength", "type": "integer"}\
        },
        "required": ["name", "age", "armor", "strength"],
        "title": "Character",
        "type": "object"
    }"""


complex_schema = """{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "Schema for a recording",
  "type": "object",
  "definitions": {
    "artist": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "functions": {
          "type": "array",
          "items": {"type": "string"}
        }
      },
      "required": ["id", "name", "functions"]
    }
  },
  "properties": {
    "id": {"type": "number"},
    "work": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "composer": {"$ref": "#/definitions/artist"}
      }
    },
    "recording_artists": {
      "type": "array",
      "items": {"$ref": "#/definitions/artist"}
    }
  },
  "required": ["id", "work", "recording_artists"]
}"""

v = Vocabulary.from_pretrained("gpt2")

import time
print("s")
st = time.time()
i = Index("A: [\w \.\*\-=\+,\?/]{10,50}\. The answer is [1-9][0-9]{0,9}\.", v)
print(time.time() - st)
print(i.get_transitions()[0])
