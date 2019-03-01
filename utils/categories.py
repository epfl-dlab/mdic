from num2words import num2words
import itertools
import regex
import spacy

nlp = spacy.load('en_core_web_lg')
forbidden = "-)("

mappings_abbrev = {
    "kg": "kilogram",
    "ml": "milliliter",
    "%": "percent",
    "lb": "pound"
}


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class SetTokens:

    def __init__(self, all_tokens, category_helper, word_to_category, category_to_word,
                 equiv=None, hierarchy_helper=None):
        self.all_tokens = all_tokens
        self.category_helper = category_helper
        self.word_to_category = word_to_category
        self.category_to_word = category_to_word
        self.hierarchy_helper = hierarchy_helper
        self.equiv = equiv

    def __repr__(self):
        return str(self.word_to_category)

    def get_occurrences(self, cat, relative=True):

        # occur = {label: len(self.category_to_word[cat][label]) for label in self.category_to_word[cat].keys()}
        occur = {label: len({self.equiv[word] for word in self.category_to_word[cat][label]})
               for label in self.category_to_word[cat].keys()}

        if relative:
            total = sum([v for k, v in occur.items()])
            occur = {k: v / total for k, v in occur.items()}

        return occur

    def from_text(self, content, verbose=True):

        # Note to my future self: a possible way to improve this would be to allow (somehow) for words to be ignored
        # in-between our token: `four diets` should match the text `four different diets`. Also conversion from lb to kg

        content_lemmatized = " ".join([token.lemma_ for token in nlp(content) if token.lemma_ not in forbidden])

        new_tokens = set()

        for t in self.all_tokens:

            if verbose:
                print(t)

            t_split = t.split(" ")

            t_expanded_split = []

            for idx, v in enumerate(t_split):

                if is_number(v):
                    tmp = num2words(float(v))
                    tmp_split = tmp.split(" ")
                    t_expanded_split += tmp_split

                elif v in mappings_abbrev:
                    t_expanded_split += [mappings_abbrev[v]]

                else:
                    t_expanded_split += [v]

            token_expanded = " ".join(t_expanded_split)

            token = regex.escape(t)
            token_expanded = regex.escape(token_expanded)

            if regex.search('(' + token + ')', content_lemmatized) is not None or \
                    regex.search('(' + token_expanded + ')', content_lemmatized) is not None:
                new_tokens.add(t)

        new_set_tokens = SetTokens.from_other_set_tokens(new_tokens, self)

        return new_set_tokens

    @staticmethod
    def from_other_set_tokens(new_tokens, other_set_token):
        category_helper = other_set_token.category_helper
        hierarchy_helper = other_set_token.hierarchy_helper
        equiv = other_set_token.equiv
        word_to_category = {k: dict() for k in category_helper.keys()}
        category_to_word = {k: {x: set() for x in category_helper[k]} for k in category_helper.keys()}

        for cat in category_helper.keys():
            for word in other_set_token.word_to_category[cat].keys():
                if word in new_tokens:
                    labels = other_set_token.word_to_category[cat][word]
                    for label in labels:
                        if word not in word_to_category[cat]:
                            word_to_category[cat][word] = set()
                        word_to_category[cat][word].add(label)
                        category_to_word[cat][label].add(word)

        return SetTokens(new_tokens, category_helper, word_to_category, category_to_word, equiv, hierarchy_helper)

    @staticmethod
    def from_annotation(ann, category_helper, hierarchy_helper=None):

        all_tokens = set()
        category_helper = category_helper
        word_to_category = {k: dict() for k in category_helper.keys()}
        category_to_word = {k: {x: set() for x in category_helper[k]} for k in category_helper.keys()}
        t_helper = dict()
        equiv = dict()

        # Create equivalence relationship helper
        with open(ann, "r") as f:
            helper_equiv = dict()
            ann_string = f.readlines()
            ann_lists = [x.strip().split("\t") for x in ann_string]
            partitions = [{x[0]} for x in ann_lists if x[0] != "*"]
            rels = [x[1].split(" ")[1:] for x in ann_lists if x[0] == "*"]

            for rel in rels:
                for src, dst in itertools.combinations(rel, 2):
                    if src not in helper_equiv:
                        helper_equiv[src] = {dst}
                    else:
                        helper_equiv[src].add(dst)

            changes = 1
            while changes != 0:
                changes, src_id, dst_id = 0, -1, -1
                for idx, set_x in enumerate(partitions):
                    for elem in set_x:
                        # print(elem)
                        if elem in helper_equiv and helper_equiv[elem].intersection(set_x) != set_x:
                            for idy, set_y in enumerate(partitions):
                                if idx != idy and len(helper_equiv[elem].intersection(set_y)) != 0:
                                    changes = 1
                                    src_id = idx
                                    dst_id = idy
                                    # print("src", partitions[src_id], "\n", "dst", partitions[dst_id])
                                    break
                            break
                    if changes == 1:
                        break

                if src_id != -1 and dst_id != -1:
                    partitions[src_id] = partitions[src_id].union(partitions[dst_id])
                    del partitions[dst_id]
            # print("end:", partitions)

            for idx, set_p in enumerate(partitions):
                for ts in set_p:
                    t_helper[ts] = idx

        # Redundantly adds labels to master
        with open(ann, "r") as f:
            ann_string = f.readlines()
            ann_lists = [x.strip().split("\t") for x in ann_string]
            ann_lists = [[x[0]] + x[1].split(" ") + [x[2]] for x in ann_lists if x[0] != "*"]
            tmp = list()
            if hierarchy_helper is not None:
                for t, label, sentence_start_span, sentence_end_span, sentence in ann_lists:
                    for master, slave in hierarchy_helper["Hierarchy"].items():
                        if label in category_helper[slave]:
                            label_master = hierarchy_helper["Hierarchy_Mapping"]["{0}-{1}".format(slave, master)][label]
                            tmp.append([t, label_master, sentence_start_span, sentence_end_span, sentence])
            ann_lists += tmp

        for t, label, sentence_start_span, sentence_end_span, sentence in ann_lists:
            sentence_lemmatized = " ".join([token.lemma_ for token in nlp(sentence) if token.lemma_ not in forbidden])
            all_tokens.add(sentence_lemmatized)
            for cat in category_helper.keys():
                if label in category_helper[cat]:
                    if sentence_lemmatized not in word_to_category[cat]:
                        word_to_category[cat][sentence_lemmatized] = set()
                    word_to_category[cat][sentence_lemmatized].add(label)
                    category_to_word[cat][label].add(sentence_lemmatized)
                    equiv[sentence_lemmatized] = t_helper[t]
                    # sentence_lemmatized
                    # equiv[sentence_lemmatized]
        return SetTokens(all_tokens, category_helper, word_to_category, category_to_word, equiv, hierarchy_helper)
