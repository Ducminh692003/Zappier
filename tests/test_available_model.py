import unittest

import orjson as json

from gemini_webapi.constants import MODEL_HEADER_KEY, Model
from gemini_webapi.types import AvailableModel
from gemini_webapi.utils import get_nested_value


def _id_for(member: Model) -> str:
    return get_nested_value(json.loads(member.model_header[MODEL_HEADER_KEY]), [4], "")


class TestAvailableModelMapping(unittest.TestCase):
    def test_model_from_name_accepts_3_1_pro_alias(self):
        self.assertIs(Model.from_name("gemini-3.1-pro"), Model.ADVANCED_PRO)
        self.assertIs(Model.from_name("gemini-3.0-pro"), Model.ADVANCED_PRO)

    def test_advanced_tier_keeps_advanced_model_names(self):
        mapping = AvailableModel.build_model_id_name_mapping(
            capacity=2,
            capacity_field=12,
        )

        self.assertEqual(
            mapping[_id_for(Model.ADVANCED_PRO)],
            Model.ADVANCED_PRO.model_name,
        )
        self.assertEqual(
            mapping[_id_for(Model.ADVANCED_FLASH)],
            Model.ADVANCED_FLASH.model_name,
        )

    def test_plus_tier_keeps_plus_model_names(self):
        mapping = AvailableModel.build_model_id_name_mapping(
            capacity=4,
            capacity_field=12,
        )

        self.assertEqual(mapping[_id_for(Model.PLUS_PRO)], Model.PLUS_PRO.model_name)
        self.assertEqual(
            mapping[_id_for(Model.PLUS_FLASH)],
            Model.PLUS_FLASH.model_name,
        )

    def test_default_mapping_preserves_basic_names(self):
        mapping = AvailableModel.build_model_id_name_mapping()

        self.assertEqual(mapping[_id_for(Model.BASIC_PRO)], Model.BASIC_PRO.model_name)
        self.assertEqual(
            mapping[_id_for(Model.BASIC_FLASH)],
            Model.BASIC_FLASH.model_name,
        )


if __name__ == "__main__":
    unittest.main()
