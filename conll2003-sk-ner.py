# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the CoNLL-2003-SK-NER Shared Task: Slovak Named Entity Recognition"""

import datasets
import json


logger = datasets.logging.get_logger(__name__)




_DESCRIPTION = """\
This is translated version of the original CONLL2003 dataset (translated from English to Slovak via Google translate) Annotation was done mostly automatically with word matching scripts. Records where some tags were not matched, were annotated manually (10%) Unlike the original Conll2003 dataset, this one contains only NER tags
"""

#_URL="/data/"
_URL = "https://github.com/ju-bezdek/conll2003-sk-ner/raw/master/data/"
_TRAINING_FILE = "train.json"
_DEV_FILE = "valid.json"
_TEST_FILE = "test.json"


class Conll2003_SK_NER_Config(datasets.BuilderConfig):
    

    def __init__(self, **kwargs):
        super(Conll2003_SK_NER_Config, self).__init__(**kwargs)


class Conll2003(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        Conll2003_SK_NER_Config(name="conll2003-SK-NER", version=datasets.Version("1.0.0"), description="Conll2003-SK-NER dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-ORG",
                                "I-ORG",
                                "B-LOC",
                                "I-LOC",
                                "B-MISC",
                                "I-MISC",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            for line in f:  
                record = json.loads(line)   
                yield guid, record
                guid += 1
            
           