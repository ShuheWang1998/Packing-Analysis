import json
import os
from tqdm import tqdm
import random
import math

random.seed(10)

# random.seed(2)


def read_jsonl(input_file):
    print(f"reading {input_file} ...")
    results = []
    with open(os.path.join(input_file), "r") as fin:
        for line in tqdm(fin):
            results.append(json.loads(line.strip()))
    
    return results


def read_json(input_file):
    return json.load(open(input_file, "r"))


def write_jsonl(data, output_file):
    file = open(output_file, "w")
    for item in tqdm(data):
        file.write(json.dumps(item, ensure_ascii=False).strip() + "\n")
    file.close()


def write_json_file(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



TAGLIST = [
    "Anthropology",
    "History",
    "Linguistics and language",
    "Philosophy",
    "Religion",
    "Economics",
    "Geography",
    "Political science",
    "Psychology",
    "Sociology",
    "Biology",
    "Chemistry",
    "Earth sciences",
    "Physics",
    "Space sciences",
    "Computer sciences",
    "Logic",
    "System science",
    "Agriculture",
    "Architecture and design",
    "Business",
    "Divinity",
    "Education",
    "Engineering and technology",
    "Chemical engineering",
    "Civil engineering",
    "Electrical engineering",
    "Materials science and engineering",
    "Mechanical engineering",
    "Environmental studies and forestry",
    "Family and consumer science",
    "Human physical performance and recreation",
    "Journalism, media studies and communication",
    "Law",
    "Library and museum studies",
    "Medicine and health",
    "Military sciences",
    "Social work",
    "Transportation",
    "Culinary arts",
    "Literature",
    "Performing arts",
    "Visual arts",
    "Mathematics",
    "Public administration",
]


def count_(name_):
    vis = {}
    num_ = 0

    transfer = {
        "Biography": "Literature",
        "Health": "Medicine and health",
        "Health and medicine": "Medicine and health",
        "Journalism media studies and communication": "Journalism, media studies and communication",
        "Media studies and communication": "Journalism, media studies and communication",
        "Statistics": "Mathematics",
        "Geometry": "Mathematics",
        "Nuclear Physics": "Physics",
        "Nuclear physics": "Physics",
        "Translation": "Linguistics and language",
        "Language and Linguistics": "Linguistics and language"
    }

    dir_ = ["/ML-A100/team/research/wangshuhe/home/opencompass/70b_hermes_evaluation/20240516_180132/predictions/llama-3-70b-instruct-hf",
            "/ML-A100/team/research/wangshuhe/home/opencompass/70b_hermes_evaluation_1/20240522_112617/predictions/llama-3-70b-instruct-hf"]

    for dir_path in dir_:
        for subdir, dirs, files in os.walk(dir_path):
            for file in files:
                if file[:4] != name_:
                    continue

                filepath = os.path.join(subdir, file)
    
                if filepath.endswith(".json"):
                    data = read_json(filepath)
                    for key, value in data.items():
                        try:
                            tag = value["prediction"].split("\"tag\": \"")[1].strip().split("\", \"explanation\":")[0].strip()

                            if tag in transfer:
                                tag = transfer[tag]
                            elif tag not in TAGLIST:
                                tag = "Others"

                            if tag not in vis:
                                vis[tag] = 0
                            vis[tag] += 1
                        except Exception as e:
                            num_ += 1
                            continue
    return vis, num_




def transfer_data(data):

    def make_prompt(item_1, item_2):
        prompt = "You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an instruction:\n[begin]\n" + item_1 + "\n[end]\nPlease choose one tag from the list below that best captures the main intent of the above instruction. Below are tags:\n[begin]\nAnthropology\n\nHistory\nLinguistics and language\nPhilosophy\nReligion\nEconomics\nGeography\nPolitical science\nPsychology\nSociology\nBiology\nChemistry\nEarth sciences\nPhysics\nSpace sciences\nComputer sciences\nLogic\nSystem science\nAgriculture\nArchitecture and design\nBusiness\nDivinity\nEducation\nEngineering and technology\nChemical engineering\nCivil engineering\nElectrical engineering\nMaterials science and engineering\nMechanical engineering\nEnvironmental studies and forestry\nFamily and consumer science\nHuman physical performance and recreation\nJournalism, media studies and communication\nLaw\nLibrary and museum studies\nMedicine and health\nMilitary sciences\nSocial work\nTransporation\nCulinary arts\nLiterature\nPerforming arts\nVisual arts\nMathematics\nPublic administration\n[end]\nYour answer should be one of the tags above and a brief explanation of your choice. Your response have to strictly follow this JSON format:[{\"tag\": str, \"explanation\": str}]. Please response in English."

        return prompt

    results = []

    for item in tqdm(data):
        for idx_ in range(0, len(item["messages"]), 2):
            instruction = item["messages"][idx_]["content"]
            answer = item["messages"][idx_+1]["content"]
            
            results.append({
                "instruction": make_prompt(item_1=instruction, item_2=answer),
                "input": "",
                "output": item["id"] + f"_{idx_}"
            })
    
    return results


def transfer_training(data):
    results = []
    for item in tqdm(data):
        messages = item["messages"]
        conversations = []
        flag = False

        if len(messages) % 2 != 0:
            continue

        for idx_ in range(0, len(messages), 2):
            if messages[idx_]["role"] == 'user':
                conversations.append({"from": "human", "value": messages[idx_]["content"]})
            else:
                flag = True
            if messages[idx_+1]["role"] == 'assistant':
                conversations.append({"from": "gpt", "value": messages[idx_+1]["content"]})
            else:
                flag = True
        if flag:
            continue

        results.append({"conversations": conversations})
    return results


def transfer_training_sharegpt(data):
    results = []
    for item in tqdm(data):
        conversations = item["conversations"]
        if len(conversations) % 2 != 0:
            print("============")
        results.append({"conversations": conversations})
    return results


def transfer_selection_data_training(data):
    choice_tag = [
        "A.",
        "a.",
        "1."
    ]

    results = []
    for item in tqdm(data):
        messages = item["messages"]

        flag = False
        for idx_ in range(0, len(messages), 2):
            for tag in choice_tag:
                if tag in messages[idx_]["content"]:
                    flag = True
                    break
            if flag:
                break
        
        if not flag:
            continue

        conversations = []
        for idx_ in range(len(messages)):
            if messages[idx_]["role"] == 'user':
                conversations.append({"from": "human", "value": messages[idx_]["content"]})
            elif messages[idx_]["role"] == 'assistant':
                conversations.append({"from": "gpt", "value": messages[idx_]["content"]})
        results.append({"conversations": conversations})
    
    print("=====")
    print(len(results))
    print("=====")

    return results


def transfer_training_equal_ratio(data, ratio_=0.001):
    multi_tag = 0
    tag_num = {}
    vis = {}
    for item in tqdm(data):
        # tag = item["tags"][0]
        # flag = True
        # if len(item["tags"]) != 1:
        #     for idx_ in range(1, len(item["tags"])):
        #         if item["tags"][idx_] != item["tags"][idx_-1]:
        #             flag = False
        #             break
        # if not flag:
        #     multi_tag += 1
        #     continue
        tag = item["tags"][0]
        sub_num = 1
        sub_vis = {}
        for idx_ in range(0, len(item["tags"])):
            if item["tags"][idx_] not in sub_vis:
                sub_vis[item["tags"][idx_]] = 0
            sub_vis[item["tags"][idx_]] += 1
            if sub_num < sub_vis[item["tags"][idx_]]:
                sub_num = sub_vis[item["tags"][idx_]]
                tag = item["tags"][idx_]

        if tag not in tag_num:
            tag_num[tag] = [0, []]
            vis[tag] = 0
        vis[tag] += 1
        tag_num[tag][0] += 1
        tag_num[tag][1].append(item)
    
    print(vis)
    # print(multi_tag)

    results = []
    for key, value in tqdm(tag_num.items()):
        # if key != "Mathematics":
        #     num_ = round(ratio_ * value[0])
        #     results = results + random.sample(value[1], num_)
        # else:
        #     results = results + value[1]
        num_ = round(ratio_ * value[0])
        results = results + random.sample(value[1], num_)
    
    return transfer_training(data=results)


def transfer_training_equal_ratio_max_bar(data, ratio_=0.001, bar=10):
    multi_tag = 0
    tag_num = {}
    vis = {}
    for item in tqdm(data):
        # tag = item["tags"][0]
        # flag = True
        # if len(item["tags"]) != 1:
        #     for idx_ in range(1, len(item["tags"])):
        #         if item["tags"][idx_] != item["tags"][idx_-1]:
        #             flag = False
        #             break
        # if not flag:
        #     multi_tag += 1
        #     continue
        tag = item["tags"][0]
        sub_num = 1
        sub_vis = {}
        for idx_ in range(0, len(item["tags"])):
            if item["tags"][idx_] not in sub_vis:
                sub_vis[item["tags"][idx_]] = 0
            sub_vis[item["tags"][idx_]] += 1
            if sub_num < sub_vis[item["tags"][idx_]]:
                sub_num = sub_vis[item["tags"][idx_]]
                tag = item["tags"][idx_]

        if tag not in tag_num:
            tag_num[tag] = [0, []]
            vis[tag] = 0
        vis[tag] += 1
        tag_num[tag][0] += 1
        tag_num[tag][1].append(item)
    
    print(vis)
    # print(multi_tag)

    results = []
    for key, value in tqdm(tag_num.items()):
        # if key != "Mathematics":
        #     num_ = round(ratio_ * value[0])
        #     results = results + random.sample(value[1], num_)
        # else:
        #     results = results + value[1]
        num_ = min(round(ratio_ * value[0]), bar)
        results = results + random.sample(value[1], num_)
    
    return transfer_training(data=results)


def transfer_training_equal_ratio_max_bar_v2(data, ratio_=0.001, bar=10):
    multi_tag = 0
    tag_num = {}
    vis = {}
    for item in tqdm(data):
        # tag = item["tags"][0]
        # flag = True
        # if len(item["tags"]) != 1:
        #     for idx_ in range(1, len(item["tags"])):
        #         if item["tags"][idx_] != item["tags"][idx_-1]:
        #             flag = False
        #             break
        # if not flag:
        #     multi_tag += 1
        #     continue
        tag = item["tags"][0]
        sub_num = 1
        sub_vis = {}
        for idx_ in range(0, len(item["tags"])):
            if item["tags"][idx_] not in sub_vis:
                sub_vis[item["tags"][idx_]] = 0
            sub_vis[item["tags"][idx_]] += 1
            if sub_num < sub_vis[item["tags"][idx_]]:
                sub_num = sub_vis[item["tags"][idx_]]
                tag = item["tags"][idx_]

        if tag not in tag_num:
            tag_num[tag] = [0, []]
            vis[tag] = 0
        vis[tag] += 1
        tag_num[tag][0] += 1
        tag_num[tag][1].append(item)
    
    print(vis)
    # print(multi_tag)

    results = []
    for key, value in tqdm(tag_num.items()):
        # if key != "Mathematics":
        #     num_ = round(ratio_ * value[0])
        #     results = results + random.sample(value[1], num_)
        # else:
        #     results = results + value[1]
        if value[0] <= bar:
            num_ = value[0]
        else:
            # print(key)
            num_ = min(round(ratio_ * value[0]), bar)
        results = results + random.sample(value[1], num_)
    
    return transfer_training(data=results)


def transfer_training_specific_tag(data, specific_tag=[""]):
    results = []
    multi_tag = 0
    for item in tqdm(data):
        # tag = item["tags"][0]
        # flag = True
        # if len(item["tags"]) != 1:
        #     for idx_ in range(1, len(item["tags"])):
        #         if item["tags"][idx_] != item["tags"][idx_-1]:
        #             flag = False
        #             break
        # if not flag:
        #     multi_tag += 1
        #     continue
        # if tag in specific_tag:
        #     results.append(item)
        flag = False
        for tag in item["tags"]:
            if tag in specific_tag:
                flag = True
                break
        if flag:
            results.append(item)
    
    # print(multi_tag)
    print(f"len: {len(results)}")
    return transfer_training(data=results)


def transfer_training_specific_tag_equal_num(data, specific_tag=[""]):
    tag_data = {}
    for item in tqdm(data):
        tag = item["tags"][0]
        flag = True
        if len(item["tags"]) != 1:
            for idx_ in range(1, len(item["tags"])):
                if item["tags"][idx_] != item["tags"][idx_-1]:
                    flag = False
                    break
        if not flag:
            continue
        if tag in specific_tag:
            if tag not in tag_data:
                tag_data[tag] = []
            tag_data[tag].append(item)
    

    min_num = 1000000000
    for key, value in tag_data.items():
        min_num = min(min_num, len(value))

    results = []
    for key, value in tag_data.items():
        results = results + random.sample(value, min_num)
    
    return transfer_training(data=results)


def transfer_training_specific_tag_specific_ratio(data, specific_tag=[""], specific_ratio=[]):
    tag_data = {}
    for item in tqdm(data):
        tag = item["tags"][0]
        flag = True
        if len(item["tags"]) != 1:
            for idx_ in range(1, len(item["tags"])):
                if item["tags"][idx_] != item["tags"][idx_-1]:
                    flag = False
                    break
        if not flag:
            continue
        if tag in specific_tag:
            if tag not in tag_data:
                tag_data[tag] = []
            tag_data[tag].append(item)
    

    min_num = 1000000000
    for key, value in tag_data.items():
        min_num = min(min_num, len(value))

    results = []
    for idx_ in range(len(specific_tag)):
        num_ = int(min_num * specific_ratio[idx_])
        results = results + random.sample(tag_data[specific_tag[idx_]], num_)
    
    return transfer_training(data=results)


def transfer_training_specific_tag_specific_ratio_base_tag_num(data, specific_tag=[""], base_tag="", specific_ratio=[]):
    tag_data = {}
    for item in tqdm(data):
        tag = item["tags"][0]
        flag = True
        if len(item["tags"]) != 1:
            for idx_ in range(1, len(item["tags"])):
                if item["tags"][idx_] != item["tags"][idx_-1]:
                    flag = False
                    break
        if not flag:
            continue
        if tag in specific_tag or "all" in specific_tag:
            if tag not in tag_data:
                tag_data[tag] = []
            tag_data[tag].append(item)
    

    min_num = len(tag_data[base_tag])

    ratio_dict = {}
    for idx_ in range(len(specific_tag)):
        ratio_dict[specific_tag[idx_]] = specific_ratio[idx_]

    results = []
    for key, value in tag_data.items():
        if key in ratio_dict:
            ratio_ = ratio_dict[key]
        else:
            ratio_ = ratio_dict["all"]
        num_ = min(int(min_num * ratio_), len(value))
        # num_ = int(min(min_num, len(value)) * ratio_)
        # print(key, num_)
        results = results + random.sample(value, num_)
    
    return transfer_training(data=results)


def transfer_training_base_specific_evaluation(data, specific_evaluations=[""], num_=10):
    tag_data = {}
    for item in tqdm(data):
        tag = item["tags"][0]
        sub_max = 0
        sub_vis_tag = {}
        for idx_ in range(len(item["tags"])):
            if item["tags"][idx_] not in sub_vis_tag:
                sub_vis_tag[item["tags"][idx_]] = 0
            sub_vis_tag[item["tags"][idx_]] += 1
            if sub_max < sub_vis_tag[item["tags"][idx_]]:
                sub_max = sub_vis_tag[item["tags"][idx_]]
                tag = item["tags"][idx_]
        if tag not in tag_data:
            tag_data[tag] = []
        tag_data[tag].append(item)
    
    evaluation_tag = {}
    for evaluation_name in specific_evaluations:
        sub_evaluation_tag, none_num = count_(name_=evaluation_name)
        print(evaluation_name, none_num)
        for key, value in sub_evaluation_tag.items():
            if key not in evaluation_tag:
                evaluation_tag[key] = 0
            evaluation_tag[key] += value

    flag = False
    if "Others" in evaluation_tag and "Others" not in tag_data:
        flag = True
    
    evaluation_num = 0
    for _, value in evaluation_tag.items():
        evaluation_num += value
    
    print(evaluation_tag)
    
    if flag:
        evaluation_num -= evaluation_tag["Others"]

    tag_index = {}
    for key, _ in tag_data.items():
        random.shuffle(tag_data[key])
        tag_index[key] = 0
    
    tag_rest = {}
    incompleted_sum = 0
    rest_num = 0
    results = []
    for key, value in evaluation_tag.items():
        if key == "Others" and flag:
            continue
        now_num = int(math.ceil(value * num_ / evaluation_num))
        if now_num > len(tag_data[key]):
            print(key, value, now_num, len(tag_data[key]))
            incompleted_sum += now_num - len(tag_data[key])
        tag_rest[key] = len(tag_data[key]) - min(now_num, len(tag_data[key]))
        if tag_rest[key] > 0:
            rest_num += 1

        # results = results + random.sample(tag_data[key], min(now_num, len(tag_data[key])))

        end_ =  min(now_num, len(tag_data[key]))
        results = results + tag_data[key][tag_index[key]:end_]
        tag_index[key] = end_
    
    for key, value in tag_data.items():
        if key not in tag_rest:
            tag_rest[key] = len(value)
            rest_num += 1
    
    # print(incompleted_sum, rest_num, span_)
    for key, value in sorted(tag_rest.items(), key=lambda item: item[1]):
        if value == 0:
            continue
        # span_ = math.ceil(incompleted_sum / rest_num)
        # now_num = span_
        # results = results + random.sample(tag_data[key], min(now_num, len(tag_data[key])))
        # rest_num -= 1
        # incompleted_sum -= min(now_num, len(tag_data[key]))
        span_ = math.ceil(incompleted_sum / rest_num)
        now_num = span_
        end_ = min(now_num + tag_index[key], len(tag_data[key]))
        results = results + tag_data[key][tag_index[key]:end_]
        rest_num -= 1
        incompleted_sum -= min(now_num, len(tag_data[key]) - tag_index[key])
        
    
    # for key, value in tag_data.items():
    #     if key not in evaluation_tag:
    #         results = results + value

    return transfer_training(data=results)


def transfer_training_without_specific_tag(data, specific_tag=[""]):
    results = []
    multi_tag = 0
    for item in tqdm(data):
        tag = item["tags"][0]
        flag = True
        if len(item["tags"]) != 1:
            for idx_ in range(1, len(item["tags"])):
                if item["tags"][idx_] != item["tags"][idx_-1]:
                    flag = False
                    break
        if not flag:
            multi_tag += 1
            continue
        if tag not in specific_tag:
            results.append(item)
    
    print(multi_tag)
    return transfer_training(data=results)


def transfer_training_less_specifc_ratio(data, specific_ratio=10):
    vis_num = {}
    for item in tqdm(data):
        for tag in item["tags"]:
            if tag not in vis_num:
                vis_num[tag] = 0
            vis_num[tag] += 1

    results = []
    for item in tqdm(data):
        flag = False
        for tag in item["tags"]:
            if vis_num[tag] <= specific_ratio:
                flag = True
                break
        if flag:
            results.append(item)
    
    return transfer_training(data=results)


def combine_single2multi_turn(data, max_turn=15, turn=10000, max_ratio=0.1):
    results = []
    vis = [idx_ for idx_ in range(len(data))]
    random.shuffle(vis)
    start_ = 0
    max_span = int(turn / (max_ratio * max_turn))
    for step in tqdm(range(turn)):
        if step % max_span == 0:
            span_ = random.randint(2, max_turn)
        else:
            span_ = random.randint(2, 3)
        conversations = []
        end_ = start_ + span_
        while start_ < end_:
            index_ = vis[start_]
            conversations = conversations + data[index_]["conversations"]
            start_ += 1
        results.append({"conversations": conversations})
    for idx_ in range(start_, len(data)):
        index_ = vis[idx_]
        results.append(data[index_])
    
    print(start_, len(data) - start_)
    print(len(results))
    return results


def combine_single2multi_turn_same_tag(orginal_data, data, max_turn=15, turn=10000, max_ratio=0.1):
    data_map = {}
    for item in orginal_data:
        content = item["messages"][0]["content"].strip() + " " + item["messages"][1]["content"].strip()
        data_map[content] = item["tags"][0]


    results = []
    vis = [idx_ for idx_ in range(len(data))]
    random.shuffle(vis)
    tag_pool = {}
    tag_pool_index = {}
    for idx_ in vis:
        content = data[idx_]["conversations"][0]["value"].strip() + " " + data[idx_]["conversations"][1]["value"].strip()
        tag = data_map[content]
        if tag not in tag_pool:
            tag_pool[tag] = []
            tag_pool_index[tag] = 0
        tag_pool[tag].append(idx_)


    index_map = {}
    start_ = 0
    max_span = int(max_ratio * max_turn)
    for step in tqdm(range(turn)):
        if step < max_span:
            span_ = random.randint(2, max_turn)
        else:
            span_ = random.randint(2, 3)
        while start_ < len(vis):
            index_ = vis[start_]
            if index_ not in index_map:
                break
            start_ += 1
        start_content = data[vis[start_]]["conversations"][0]["value"].strip() + " " + data[vis[start_]]["conversations"][1]["value"].strip()
        start_tag = data_map[start_content]

        end_ = tag_pool_index[start_tag] + span_
        conversations = []
        while tag_pool_index[start_tag] < end_ and tag_pool_index[start_tag] < len(tag_pool[start_tag]):
            index_ = tag_pool[start_tag][tag_pool_index[start_tag]]
            conversations = conversations + data[index_]["conversations"]
            index_map[index_] = True
            tag_pool_index[start_tag] += 1
        results.append({"conversations": conversations})
    
    print(len(results))
        
    single_num = 0
    for idx_ in range(len(data)):
        index_ = vis[idx_]
        if index_ not in index_map:
            results.append(data[index_])
            single_num += 1
    
    print(len(data) - single_num, single_num)
    print(len(results))
    return results


def multi_turn_plus_single_turn(data):
    results = []
    for item in tqdm(data):
        results.append(item)
        if len(item["conversations"]) == 2:
            continue
        for idx_ in range(0, len(item["conversations"]), 2):
            conversation = []
            conversation.append(item["conversations"][idx_])
            conversation.append(item["conversations"][idx_+1])
            results.append({"conversations": conversation})
        
    return results


def extract_multi_turn(data, num_=10):
    results = []
    for item in tqdm(data):
        if len(item["conversations"]) > 2:
            results.append(item)

    if num_ == -1:
        return results
    return random.sample(results, num_)


def extract_single_turn(data, num_=10):
    results = []
    for item in tqdm(data):
        if len(item["conversations"]) == 2:
            results.append(item)

    if num_ == -1:
        return results
    return random.sample(results, num_)


def extract_single_turn_from_multi(data):
    results = []
    for item in tqdm(data):
        if len(item["conversations"]) > 2:
            for idx_ in range(0, len(item["conversations"]), 2):
                conversation = []
                conversation.append(item["conversations"][idx_])
                conversation.append(item["conversations"][idx_+1])
                results.append({"conversations": conversation})
    return results


def combine_loss_data(full_data, loss_data):
    id2index = {}
    for idx_, item in tqdm(enumerate(full_data)):
        id2index[item["id"]] = idx_
    
    for item in loss_data:
        full_data[id2index[item["id"]]] = item
    
    return full_data


if __name__ == '__main__':
    # for idx_ in range(7):
    #     hermes = read_jsonl(input_file=f"/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/data_distribution/{idx_}/{idx_}.jsonl")
    #     formatted_data = transfer_data(data=hermes)
    #     write_jsonl(data=formatted_data, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/data_distribution_hermes_{idx_}.jsonl")
    
    # open_instruct_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5.jsonl")
    # formatted_hermes = transfer_training(data=open_instruct_hermes)
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_training.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio(data=tag_hermes, ratio_=0.3)
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_3_math.json")
    # name = {
    #     "Linguistics and language": "language",
    #     "History": "history",
    #     "Computer sciences": "computer",
    #     "Education": "education",
    #     "Literature": "literature",
    #     "Mathematics": "mathematics"
    # }
    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # for specifci_tag in ["Linguistics and language", "History", "Computer sciences", "Education", "Literature", "Mathematics"]:
    #     formatted_hermes = transfer_training_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    #     write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_{name[specifci_tag]}.json")
    # name = {
    #     "Linguistics and language": "language",
    #     "History": "history",
    #     "Computer sciences": "computer",
    #     "Education": "education",
    #     "Literature": "literature",
    #     "Mathematics": "mathematics"
    # }
    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # for specifci_tag in ["Linguistics and language", "History", "Computer sciences", "Education", "Literature", "Mathematics"]:
    #     formatted_hermes = transfer_training_without_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    #     write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_without_tag_{name[specifci_tag]}.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Linguistics and language", "History", "Computer sciences", "Education", "Literature", "Mathematics"]
    # formatted_hermes = transfer_training_without_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_without_tag_top6.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # # specifci_tag = ["Computer sciences", "Mathematics"]
    # specifci_tag = ["Linguistics and language", "Education", "Literature", "History", "Mathematics"]
    # formatted_hermes = transfer_training_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_language_education_literature_history_math.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Computer sciences", "Mathematics"]
    # formatted_hermes = transfer_training_specific_tag_equal_num(data=tag_hermes, specific_tag=specifci_tag)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_equal_computer_math.json")

    # meta_math = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/Meta_Math/meta_math_new_query.jsonl")
    # formatted_math = transfer_training(data=meta_math)
    # write_json_file(data=formatted_math, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_training.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Computer sciences", "Mathematics"]
    # specific_ratio = [0.5, 1]
    # formatted_hermes = transfer_training_specific_tag_specific_ratio(data=tag_hermes, specific_tag=specifci_tag, specific_ratio=specific_ratio)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_computer1_math2.json")

    # meta_math_only_math = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/Meta_Math/meta_math_new_query_only_math.jsonl")
    # formatted_math = transfer_training(data=meta_math_only_math)
    # write_json_file(data=formatted_math, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_only_math_training.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Computer sciences", "Mathematics", "all"]
    # specific_ratio = [1, 1, 0.05]
    # formatted_hermes = transfer_training_specific_tag_specific_ratio_base_tag_num(data=tag_hermes, specific_tag=specifci_tag, base_tag="Mathematics", specific_ratio=specific_ratio)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_computer1_math1_all005.json")

    # meta_math_without_gsm_format = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/Meta_Math/meta_math_new_query_without_gsm_format.jsonl")
    # formatted_math = transfer_training(data=meta_math_without_gsm_format)
    # write_json_file(data=formatted_math, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_new_query_without_gsm_format_training.json")

    # tag_meta_math = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/Meta_Math/meta_math_new_query_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Mathematics"]
    # formatted_hermes = transfer_training_specific_tag(data=tag_meta_math, specific_tag=specifci_tag)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_math.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Philosophy", "Law", "Biology", "Psychology", "Medicine and health", "Economics", "History", "Mathematics"]
    # formatted_hermes = transfer_training_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_philosophy_law_biology_psychology_medicine_economics_history_math.json")

    # only_computer_math1 = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_only_computer_computer1_math1.json")
    # mmlu_data_math = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_philosophy_law_biology_psychology_medicine_economics_history_math.json")
    # formatted_data = only_computer_math1+mmlu_data_math
    # print(len(formatted_data))
    # write_json_file(data=formatted_data, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_philosophy_law_biology_psychology_medicine_economics_history_computermath1_math.json")

    # meta_math_150K = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/Meta_Math/150k_meta_math_new_query.jsonl")
    # formatted_math = transfer_training(data=meta_math_150K)
    # write_json_file(data=formatted_math, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_training_150k.json")

    # meta_math_150k = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_training.json")
    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Linguistics and language", "History", "Computer sciences", "Education", "Literature"]
    # formatted_hermes = transfer_training_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    # write_json_file(data=meta_math_150k+formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_hermes_top5.json")

    # meta_math_official_format = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/Meta_Math/meta_math_new_query_official_format.jsonl")
    # formatted_math = transfer_training(data=meta_math_official_format)
    # write_json_file(data=formatted_math, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_official_prompt_training.json")

    # open_instruct_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5.jsonl")
    # formatted_hermes = transfer_selection_data_training(data=open_instruct_hermes)
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_loose_choice_question_training.json")

    # meta_math_official_format = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_official_prompt_training.json")
    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Linguistics and language", "History", "Computer sciences", "Education", "Literature"]
    # formatted_hermes = transfer_training_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    # print(len(meta_math_official_format+formatted_hermes))
    # write_json_file(data=meta_math_official_format+formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_official_prompt_hermes_top5.json")

    # meta_math_official_format = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_official_prompt_training.json")
    # hermes_math_tag = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_mathematics.json")
    # print(len(meta_math_official_format + hermes_math_tag))
    # write_json_file(data=meta_math_official_format+hermes_math_tag, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/meta_math_official_prompt_hermes_math.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Mathematics", "Computer sciences", "Linguistics and language", "Literature", "Biology", "History"]
    # formatted_hermes = transfer_training_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    # write_json_file(data=formatted_hermes, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_top_6.json")

    # tulu_v2 = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/tulu_v2/tulu_v1_data.jsonl")
    # formatted_tulu = transfer_training(data=tulu_v2)
    # write_json_file(data=formatted_tulu, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specifci_tag = ["Mathematics"]
    # formatted_all = transfer_training_specific_tag(data=all_data, specific_tag=specifci_tag)
    # write_json_file(data=formatted_all, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_math.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specifci_tag = ["Computer sciences"]
    # formatted_all = transfer_training_specific_tag(data=all_data, specific_tag=specifci_tag)
    # write_json_file(data=formatted_all, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_computer.json")

    # wild_chat = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/WildChat/wild_chat_only_gpt_4.jsonl")
    # formatted_wildchat = transfer_training(data=wild_chat)
    # write_json_file(data=formatted_wildchat, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")

    # wild_chat_single = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/WildChat/wild_chat_only_gpt_4_single_turn.jsonl")
    # formatted_wildchat = transfer_training(data=wild_chat_single)
    # write_json_file(data=formatted_wildchat, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training_coarse_single_turn.json")

    # hermes_200_single = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2.json")
    # multi_turn = combine_single2multi_turn(data=hermes_200_single, max_turn=10, turn=50000, max_ratio=0.01)
    # write_json_file(data=multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # hermes_200_single = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2.json")
    # multi_turn = combine_single2multi_turn_same_tag(orginal_data=tag_hermes, data=hermes_200_single, max_turn=10, turn=50000, max_ratio=0.01)
    # write_json_file(data=multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag.json")

    # open_instruct_yi_132 = read_jsonl(input_file="/ML-A100/team/research/wangshuhe/home/opencompass/70b_yi_100k/tag_data/all_yi_100k.jsonl")
    # formatted_yi_132 = transfer_training(data=open_instruct_yi_132)
    # write_json_file(data=formatted_yi_132, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/yi_132_training.json")

    # all_without_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # formatted_all_without_hermes = transfer_training(data=all_without_hermes)
    # span_ = [(0, 400000), (400000, 800000), (800000, -1)]
    # for idx_ in range(3):
    #     write_json_file(data=formatted_all_without_hermes[span_[idx_][0]:span_[idx_][1]], output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/aya_chatarena_lima_metamath_norobots_sharegpt_ultrachat200_wildchat_wizard_tuluv2/all_data_without_hermes_{idx_}.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # formatted_all_data = transfer_training(data=all_data)
    # span_ = [(0, 1000000), (1000000, 1500000), (1500000, 1800000), (1800000, -1)]
    # for idx_ in range(4):
    #     write_json_file(data=formatted_all_data[span_[idx_][0]:span_[idx_][1]], output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2/all_data_{idx_}.json")

    # ultra_chat_200k = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/UltraChat/ultrachat_200K.jsonl")
    # formatted_ultra_chat_200k = transfer_training(data=ultra_chat_200k)
    # write_json_file(data=formatted_ultra_chat_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/ultrachat_200k_training.json")

    # open_instruct_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5.jsonl")
    # formatted_hermes = transfer_training(data=open_instruct_hermes)
    # write_json_file(data=formatted_hermes+formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_training_test.json")

    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # multi_turn_and_single = multi_turn_plus_single_turn(data=hermes_200_multi_turn)
    # print(len(multi_turn_and_single))
    # write_json_file(data=multi_turn_and_single, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_plus_single_turn.json")

    # hermes_200_multi_turn_same_tag = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag.json")
    # multi_turn_and_single = multi_turn_plus_single_turn(data=hermes_200_multi_turn_same_tag)
    # print(len(multi_turn_and_single))
    # write_json_file(data=multi_turn_and_single, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag_single_turn.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn_same_tag = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag.json")
    # print(len(hermes_200_multi_turn_same_tag))
    # print(len(wild_chat_training+hermes_200_multi_turn_same_tag))
    # write_json_file(data=wild_chat_training+hermes_200_multi_turn_same_tag, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag_wild_chat.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # print(len(hermes_200_multi_turn))
    # print(len(wild_chat_training+hermes_200_multi_turn))
    # write_json_file(data=wild_chat_training+hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_wild_chat.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn_same_tag = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag.json")
    # only_multi_turn_hermes_200_multi_turn_same_tag = extract_multi_turn(data=hermes_200_multi_turn_same_tag)
    # print(len(wild_chat_training+only_multi_turn_hermes_200_multi_turn_same_tag))
    # write_json_file(data=wild_chat_training+only_multi_turn_hermes_200_multi_turn_same_tag, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_multi_turn_same_tag_wild_chat.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_multi_turn_hermes_200_multi_turn = extract_multi_turn(data=hermes_200_multi_turn)
    # print(len(wild_chat_training+only_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=wild_chat_training+only_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_multi_turn_wild_chat.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200k = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2.json")
    # print(len(wild_chat_training + hermes_200k))
    # write_json_file(data=wild_chat_training+hermes_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_wild_chat.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn_same_tag = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag.json")
    # only_single_turn_hermes_200_multi_turn_same_tag = extract_single_turn(data=hermes_200_multi_turn_same_tag)
    # print(len(wild_chat_training+only_single_turn_hermes_200_multi_turn_same_tag))
    # write_json_file(data=wild_chat_training+only_single_turn_hermes_200_multi_turn_same_tag, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_single_multi_turn_same_tag_wild_chat.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn_same_tag = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn_same_tag.json")
    # only_single_turn_hermes_200_multi_turn_same_tag = extract_single_turn_from_multi(data=hermes_200_multi_turn_same_tag)
    # print(len(wild_chat_training+only_single_turn_hermes_200_multi_turn_same_tag))
    # write_json_file(data=wild_chat_training+only_single_turn_hermes_200_multi_turn_same_tag, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_single_from_multi_multi_turn_same_tag_wild_chat.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_multi_turn_hermes_200_multi_turn = extract_multi_turn(data=hermes_200_multi_turn)
    # print(len(tulu_training+only_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=tulu_training+only_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_multi_turn_tulu.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_multi_turn_hermes_200_multi_turn = random.sample(extract_multi_turn(data=hermes_200_multi_turn), 25000)
    # print(len(wild_chat_training+only_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=wild_chat_training+only_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_half_multi_turn_wild_chat.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_multi_turn_hermes_200_multi_turn = random.sample(extract_multi_turn(data=hermes_200_multi_turn), 10000)
    # print(len(wild_chat_training+only_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=wild_chat_training+only_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_10K_multi_turn_wild_chat.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_single_from_multi_turn_hermes_200_multi_turn = extract_single_turn_from_multi(data=hermes_200_multi_turn)
    # print(len(tulu_training+only_single_from_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=tulu_training+only_single_from_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_single_from_multi_multi_turn_tulu.json")

    # hermes = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_training.json")
    # multi_turn = combine_single2multi_turn(data=hermes, max_turn=30, turn=100000, max_ratio=0.01)
    # write_json_file(data=multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_100k.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # hermes_multi_turn_100k = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_100k.json")
    # only_multi_turn_hermes_multi_turn_100k = extract_multi_turn(data=hermes_multi_turn_100k)
    # print(len(tulu_training+only_multi_turn_hermes_multi_turn_100k))
    # write_json_file(data=tulu_training+only_multi_turn_hermes_multi_turn_100k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_only_multi_multi_turn_100k_tulu.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # formatted_all = transfer_training_equal_ratio(data=all_data, ratio_=0.5)
    # span_ = [(0, 500000), (500000, -1)]
    # for idx_ in range(2):
    #     write_json_file(data=formatted_all[span_[idx_][0]:span_[idx_][1]], output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_ratio_0_5/all_data_tag_ratio_0_5_{idx_}.json")

    # tulu = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_random_100K = extract_single_turn(data=tulu, num_=100000)
    # write_json_file(data=tulu_random_100K, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k.json")

    # tulu_random_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k.json")
    # hermes_multi_turn_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_100k.json")
    # hermes_multi_turn_100K_random_500 = extract_multi_turn(data=hermes_multi_turn_100K, num_=500)
    # print(len(tulu_random_100K + hermes_multi_turn_100K_random_500))
    # write_json_file(data=tulu_random_100K + hermes_multi_turn_100K_random_500, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k_fake_multi_turn_500.json")

    # tulu_random_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k.json")
    # hermes_multi_turn_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_100k.json")
    # hermes_multi_turn_100K_random_1k = extract_multi_turn(data=hermes_multi_turn_100K, num_=1000)
    # print(len(tulu_random_100K + hermes_multi_turn_100K_random_1k))
    # write_json_file(data=tulu_random_100K + hermes_multi_turn_100K_random_1k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k_fake_multi_turn_1k.json")

    # tulu_random_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k.json")
    # hermes_multi_turn_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_100k.json")
    # hermes_multi_turn_100K_random_2k = extract_multi_turn(data=hermes_multi_turn_100K, num_=2000)
    # print(len(tulu_random_100K + hermes_multi_turn_100K_random_2k))
    # write_json_file(data=tulu_random_100K + hermes_multi_turn_100K_random_2k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k_fake_multi_turn_2k.json")

    # tulu_random_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k.json")
    # wildchat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wildchat_multi_turn_random_500 = extract_multi_turn(data=wildchat, num_=500)
    # print(len(tulu_random_100K + wildchat_multi_turn_random_500))
    # write_json_file(data=tulu_random_100K + wildchat_multi_turn_random_500, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k_true_multi_turn_500.json")

    # tulu_random_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k.json")
    # wildchat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wildchat_multi_turn_random_1k = extract_multi_turn(data=wildchat, num_=1000)
    # print(len(tulu_random_100K + wildchat_multi_turn_random_1k))
    # write_json_file(data=tulu_random_100K + wildchat_multi_turn_random_1k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k_true_multi_turn_1k.json")

    # tulu_random_100K = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k.json")
    # wildchat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wildchat_multi_turn_random_2k = extract_multi_turn(data=wildchat, num_=2000)
    # print(len(tulu_random_100K + wildchat_multi_turn_random_2k))
    # write_json_file(data=tulu_random_100K + wildchat_multi_turn_random_2k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_single_turn_random_100k_true_multi_turn_2k.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # less_than_50k = transfer_training_less_specifc_ratio(data=all_data, specific_ratio=50000)
    # print(len(less_than_50k))
    # write_json_file(data=less_than_50k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_less_than_50k.jsonl")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specifci_tag = ["Linguistics and language"]
    # formatted_all = transfer_training_specific_tag(data=all_data, specific_tag=specifci_tag)
    # write_json_file(data=formatted_all, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_language.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specifci_tag = ["Literature"]
    # formatted_all = transfer_training_specific_tag(data=all_data, specific_tag=specifci_tag)
    # write_json_file(data=formatted_all, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_literature.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_multi_turn_hermes_200_multi_turn = extract_multi_turn(data=hermes_200_multi_turn, num_=25000)
    # print(len(tulu_training+only_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=tulu_training+only_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_half_multi_turn_tulu.json")

    # share_gpt = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/ShareGPT/share_gpt_4096.jsonl")
    # formatted_share_gpt = transfer_training(data=share_gpt)
    # write_json_file(data=formatted_share_gpt, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/share_gpt_4096_training.json")

    # share_gpt_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/share_gpt_4096_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_multi_turn_hermes_200_multi_turn = extract_multi_turn(data=hermes_200_multi_turn, num_=-1)
    # print(len(share_gpt_training+only_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=share_gpt_training+only_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_multi_turn_sharegpt_4096.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_ratio_new.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka", "bbh-", "math", "gsm8"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_bbh_math_gsm8k_ratio.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka"]
    # all_data_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(all_data_mmlu_ratio_200k))
    # write_json_file(data=all_data_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_tag_200k_mmlu_ratio.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # formatted_all = transfer_training_equal_ratio(data=all_data, ratio_=0.093)
    # print(len(formatted_all))
    # write_json_file(data=formatted_all, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_equal_ratio_200k.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka", "bbh-", "math", "gsm8"]
    # all_data_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(all_data_mmlu_ratio_200k))
    # write_json_file(data=all_data_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_200k_mmlu_bbh_math_gsm8k_ratio.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["alpa"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_alpa_farm_ratio.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["alpa"]
    # all_data_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(all_data_mmlu_ratio_200k))
    # write_json_file(data=all_data_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_tag_200k_alpa_ratio.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["IFEv"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_ifeval_ratio.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka", "bbh-", "gsm8"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_bbh_gsm8k_ratio.json")

    # hermes = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_training.json")
    # multi_turn = combine_single2multi_turn(data=hermes, max_turn=30, turn=150000, max_ratio=0.01)
    # write_json_file(data=multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_150k.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # hermes_multi_turn_150k = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_150k.json")
    # only_multi_turn_hermes_multi_turn_150k = extract_multi_turn(data=hermes_multi_turn_150k, num_=-1)
    # print(len(tulu_training))
    # print(len(only_multi_turn_hermes_multi_turn_150k))
    # print(len(tulu_training+only_multi_turn_hermes_multi_turn_150k))
    # write_json_file(data=tulu_training+only_multi_turn_hermes_multi_turn_150k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_only_multi_multi_turn_150k_tulu.json")

    # share_gpt_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/share_gpt_4096_training.json")
    # hermes_multi_turn_100k = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_multi_turn_100k.json")
    # only_multi_turn_hermes_multi_turn_100k = extract_multi_turn(data=hermes_multi_turn_100k, num_=-1)
    # print(len(share_gpt_training+only_multi_turn_hermes_multi_turn_100k))
    # write_json_file(data=share_gpt_training+only_multi_turn_hermes_multi_turn_100k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_only_multi_multi_turn_100k_sharegpt_4096.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["IFEv"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_200k_ifeval_ratio.json")
    
    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka", "bbh-", "gsm8", "IFEv"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_bbh_gsm8k_ifeval_ratio.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka"]
    # all_data_mmlu_ratio_300k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=300000)
    # print(len(all_data_mmlu_ratio_300k))
    # write_json_file(data=all_data_mmlu_ratio_300k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_tag_300k_mmlu_ratio.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # formatted_all = transfer_training_equal_ratio(data=all_data, ratio_=0.1385)
    # print(len(formatted_all))
    # write_json_file(data=formatted_all, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_equal_ratio_300k.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # formatted_all = transfer_training_equal_ratio(data=all_data, ratio_=0.093)
    # print(len(formatted_all))
    # write_json_file(data=formatted_all, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_equal_ratio_200k_seed_32.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # formatted_all = transfer_training_equal_ratio(data=all_data, ratio_=0.1845)
    # print(len(formatted_all))
    # write_json_file(data=formatted_all, output_file=f"/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_equal_ratio_400k.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=5000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(wild_chat + only_multi_turn))
    # write_json_file(data=wild_chat + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_5k.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=5000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # wild_chat_only_multi = extract_multi_turn(data=wild_chat, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(wild_chat_only_multi), len(only_single + only_multi_turn + wild_chat_only_multi))
    # write_json_file(data=only_single + only_multi_turn + wild_chat_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_5k_without_same_single.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio_max_bar(data=tag_hermes, ratio_=0.246, bar=30000)
    # print(len(formatted_hermes))
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_max_30K.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio(data=tag_hermes, ratio_=0.2)
    # print(len(formatted_hermes))
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_seed_32.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # hermes_200_single = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2.json")
    # print(len(tulu_training+hermes_200_single))
    # write_json_file(data=tulu_training+hermes_200_single, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_tulu.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["IFEv"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_ifeval_ratio_seed_2.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio_max_bar(data=tag_hermes, ratio_=0.246, bar=30000)
    # print(len(formatted_hermes))
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_max_30K_seed_16.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka", "bbh-", "gsm8"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_bbh_gsm8k_ratio_seed_16.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=25000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(tulu_training + only_multi_turn))
    # write_json_file(data=tulu_training + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_25k.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=25000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # tulu_only_multi = extract_multi_turn(data=tulu_training, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(tulu_only_multi), len(only_single + only_multi_turn + tulu_only_multi))
    # write_json_file(data=only_single + only_multi_turn + tulu_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_25k_without_same_single.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka", "bbh-", "gsm8"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_bbh_gsm8k_ratio_seed_2.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio_max_bar(data=tag_hermes, ratio_=0.303, bar=20000)
    # print(len(formatted_hermes))
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_max_20K.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio_max_bar(data=tag_hermes, ratio_=0.478, bar=10000)
    # print(len(formatted_hermes))
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_max_10K.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio_max_bar(data=tag_hermes, ratio_=0.202, bar=60000)
    # print(len(formatted_hermes))
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_max_60K.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # formatted_hermes = transfer_training_equal_ratio_max_bar_v2(data=tag_hermes, ratio_=0.115, bar=13000)
    # print(len(formatted_hermes))
    # write_json_file(data=formatted_hermes, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_max_13K_v2.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=10000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(wild_chat + only_multi_turn))
    # write_json_file(data=wild_chat + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_10k.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=10000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # wild_chat_only_multi = extract_multi_turn(data=wild_chat, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(wild_chat_only_multi), len(only_single + only_multi_turn + wild_chat_only_multi))
    # write_json_file(data=only_single + only_multi_turn + wild_chat_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_10k_without_same_single.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=10000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(tulu_training + only_multi_turn))
    # write_json_file(data=tulu_training + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_10k.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=10000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # tulu_only_multi = extract_multi_turn(data=tulu_training, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(tulu_only_multi), len(only_single + only_multi_turn + tulu_only_multi))
    # write_json_file(data=only_single + only_multi_turn + tulu_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_10k_without_same_single.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=12000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(wild_chat + only_multi_turn))
    # write_json_file(data=wild_chat + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_12k.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=12000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # wild_chat_only_multi = extract_multi_turn(data=wild_chat, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(wild_chat_only_multi), len(only_single + only_multi_turn + wild_chat_only_multi))
    # write_json_file(data=only_single + only_multi_turn + wild_chat_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_12k_without_same_single.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka", "bbh-", "gsm8"]
    # all_data_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(all_data_mmlu_ratio_200k))
    # write_json_file(data=all_data_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_200k_mmlu_bbh_gsm8k_ratio.json")

    # magpie_pro_llama = read_jsonl(input_file="/ML-A100/team/research/magpie_dataset/Magpie-Align/Magpie-Pro-300K-Filtered/train.jsonl")
    # formatted_magpie_pro_llama = transfer_training_sharegpt(data=magpie_pro_llama)
    # print(len(formatted_magpie_pro_llama))
    # write_json_file(data=formatted_magpie_pro_llama, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/magpie_pro_llama_training.json")

    # magpie_pro_qwen2 = read_jsonl(input_file="/ML-A100/team/research/magpie_dataset/Magpie-Align/Magpie-Qwen2-Pro-300K-Filtered/train.jsonl")
    # formatted_magpie_pro_qwen2 = transfer_training_sharegpt(data=magpie_pro_qwen2)
    # print(len(formatted_magpie_pro_qwen2))
    # write_json_file(data=formatted_magpie_pro_qwen2, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/magpie_pro_qwen2_training.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka", "gsm8", "IFEv"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_gsm8k_ifeval_ratio.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka", "gsm8", "IFEv"]
    # all_data_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(all_data_mmlu_ratio_200k))
    # write_json_file(data=all_data_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_200k_mmlu_gsm8k_ifeval_ratio.json")

    # magpie_pro_llama = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/magpie_pro_llama_training.json")
    # multi_turn = combine_single2multi_turn(data=magpie_pro_llama, max_turn=10, turn=10000, max_ratio=0.01)
    # print(len(multi_turn))
    # write_json_file(data=multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/magpie_pro_llama_self_fake_multi_turn_10k_without_same_single.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=3000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(wild_chat + only_multi_turn))
    # write_json_file(data=wild_chat + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_3k.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=3000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # wild_chat_only_multi = extract_multi_turn(data=wild_chat, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(wild_chat_only_multi), len(only_single + only_multi_turn + wild_chat_only_multi))
    # write_json_file(data=only_single + only_multi_turn + wild_chat_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_3k_without_same_single.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=1000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(wild_chat + only_multi_turn))
    # write_json_file(data=wild_chat + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_1k.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=1000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # wild_chat_only_multi = extract_multi_turn(data=wild_chat, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(wild_chat_only_multi), len(only_single + only_multi_turn + wild_chat_only_multi))
    # write_json_file(data=only_single + only_multi_turn + wild_chat_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_1k_without_same_single.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka", "bbh-", "gsm8"]
    # all_data_mmlu_ratio_300k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=300000)
    # print(len(all_data_mmlu_ratio_300k))
    # write_json_file(data=all_data_mmlu_ratio_300k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_300k_mmlu_bbh_gsm8k_ratio.json")

    # tag_hermes = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specific_evaluations = ["luka", "gsm8", "IFEv", "IFEv"]
    # hermes_mmlu_ratio_200k = transfer_training_base_specific_evaluation(data=tag_hermes, specific_evaluations=specific_evaluations, num_=200000)
    # print(len(hermes_mmlu_ratio_200k))
    # write_json_file(data=hermes_mmlu_ratio_200k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_200k_mmlu_gsm8k_2_ifeval_ratio.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=5000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(tulu_training + only_multi_turn))
    # write_json_file(data=tulu_training + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_5k.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=5000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # tulu_only_multi = extract_multi_turn(data=tulu_training, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(tulu_only_multi), len(only_single + only_multi_turn + tulu_only_multi))
    # write_json_file(data=only_single + only_multi_turn + tulu_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_5k_without_same_single.json")

    # wild_chat_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200_multi_turn = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_multi_turn.json")
    # only_multi_turn_hermes_200_multi_turn = extract_multi_turn(data=hermes_200_multi_turn, num_=5000)
    # print(len(wild_chat_training+only_multi_turn_hermes_200_multi_turn))
    # write_json_file(data=wild_chat_training+only_multi_turn_hermes_200_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_only_multi_5K_multi_turn_wild_chat.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=50, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(wild_chat + only_multi_turn))
    # write_json_file(data=wild_chat + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_50.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=50, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # wild_chat_only_multi = extract_multi_turn(data=wild_chat, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(wild_chat_only_multi), len(only_single + only_multi_turn + wild_chat_only_multi))
    # write_json_file(data=only_single + only_multi_turn + wild_chat_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_50_without_same_single.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=18000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(wild_chat + only_multi_turn))
    # write_json_file(data=wild_chat + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_18k.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=18000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # wild_chat_only_multi = extract_multi_turn(data=wild_chat, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(wild_chat_only_multi), len(only_single + only_multi_turn + wild_chat_only_multi))
    # write_json_file(data=only_single + only_multi_turn + wild_chat_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_18k_without_same_single.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=90000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # print(len(tulu_training + only_multi_turn))
    # write_json_file(data=tulu_training + only_multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_90k.json")

    # tulu_training = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_training.json")
    # tulu_single_turn = extract_single_turn(data=tulu_training, num_=-1)
    # multi_turn = combine_single2multi_turn(data=tulu_single_turn, max_turn=10, turn=90000, max_ratio=0.01)
    # only_multi_turn = extract_multi_turn(data=multi_turn, num_=-1)
    # only_single = extract_single_turn(data=multi_turn, num_=-1)
    # tulu_only_multi = extract_multi_turn(data=tulu_training, num_=-1)
    # print(len(only_single), len(only_multi_turn), len(tulu_only_multi), len(only_single + only_multi_turn + tulu_only_multi))
    # write_json_file(data=only_single + only_multi_turn + tulu_only_multi, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/tulu_self_fake_multi_turn_90k_without_same_single.json")

    # lima = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/LIMA/lima.jsonl")
    # formatted_lima = transfer_training(data=lima)
    # write_json_file(data=formatted_lima, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/lima_training.json")

    # data = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # print(len(data))
    # data = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_self_fake_multi_turn_1k_without_same_single.json")
    # print(len(data))

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # print(len(wild_chat_single_turn))
    # write_json_file(data=wild_chat_single_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_only_single.json")

    # wild_chat = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_training.json")
    # wild_chat_single_turn = extract_single_turn(data=wild_chat, num_=-1)
    # multi_turn = combine_single2multi_turn(data=wild_chat_single_turn, max_turn=10, turn=18000, max_ratio=0.01)
    # print(len(multi_turn))
    # write_json_file(data=multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/wild_chat_fake_multi_from_only_single.json")

    # hermes_200_single = read_json(input_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2.json")
    # multi_turn = combine_single2multi_turn(data=hermes_200_single, max_turn=10, turn=79000, max_ratio=0.01)
    # print(len(multi_turn))
    # write_json_file(data=multi_turn, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/hermes_tag_ratio_0_2_full_self_multi_turn.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka", "gsm8", "IFEv"]
    # all_data_mmlu_ratio_300k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=300000)
    # print(len(all_data_mmlu_ratio_300k))
    # write_json_file(data=all_data_mmlu_ratio_300k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_300k_mmlu_gsm8k_ifeval_ratio.json")

    # all_data = read_jsonl(input_file="/ML-A800/home/wangshuhe/open-instruct/data/processed/all_tag_data/aya_chatarena_lima_metamath_norobots_hermes_sharegpt_ultrachat200_wildchat_wizard_tuluv2.jsonl")
    # specific_evaluations = ["luka", "gsm8", "IFEv", "IFEv", "IFEv", "IFEv", "IFEv"]
    # all_data_mmlu_ratio_300k = transfer_training_base_specific_evaluation(data=all_data, specific_evaluations=specific_evaluations, num_=300000)
    # print(len(all_data_mmlu_ratio_300k))
    # write_json_file(data=all_data_mmlu_ratio_300k, output_file="/ML-A100/team/research/wangshuhe/home/LLaMA-Factory/data/all_data_tag_300k_mmlu_gsm8k_ifeval_5_ratio.json")

    # original_data = read_jsonl(input_file="/xpfs/wangshuhe/A800/open-instruct/data/processed/WildChat/wild_chat_only_gpt_4.jsonl")
    # loss_data = read_jsonl(input_file="/xpfs/wangshuhe/A100/home/prompt2ppl/open-instruct/output_dir/wildchat/gpt_4/full/full_gpt_4_loss_data.jsonl")
    # combined_data = combine_loss_data(full_data=original_data, loss_data=loss_data)
    # print(len(combined_data))
    # write_json_file(data=transfer_training(combined_data), output_file="/xpfs/wangshuhe/A100/home/LLaMA-Factory/data/wild_chat_training_gpt_4_full_loss.json")
    
    # self_paraphrase_data = read_jsonl(input_file="/xpfs/wangshuhe/A100/home/prompt2ppl/open-instruct/output_dir/tulu/full_paraphrase_reanswer.jsonl")
    # print(len(self_paraphrase_data))
    # write_json_file(data=transfer_training(self_paraphrase_data), output_file="/xpfs/wangshuhe/A100/home/LLaMA-Factory/data/tulu_training_self_paraphrase_reanswer.json")

    # wildchat_full = read_jsonl(input_file="/xpfs/wangshuhe/A800/open-instruct/data/processed/WildChat/wild_chat.jsonl")
    # formatted_wildchat_full = transfer_training(data=wildchat_full)
    # write_json_file(data=formatted_wildchat_full, output_file="/xpfs/wangshuhe/A100/home/LLaMA-Factory/data/wildchat_full_training.json")

    # wildchat_full = read_jsonl(input_file="/xpfs/wangshuhe/A800/open-instruct/data/processed/WildChat/wild_chat.jsonl")
    # formatted_wildchat_full = transfer_training(data=wildchat_full)
    # span_ = [(0, 200000), (200000, 400000), (400000, -1)]
    # for idx_ in range(3):
    #     write_json_file(data=formatted_wildchat_full[span_[idx_][0]:span_[idx_][1]], output_file=f"/xpfs/wangshuhe/A100/home/LLaMA-Factory/data/wildchat_full/wildchat_full_{idx_}.json")

    # wild_chat_training = read_json(input_file="/gpfs/wangshuhe/A100/home/LLaMA-Factory/data/wild_chat_training.json")
    # hermes_200k = read_json(input_file="/gpfs/wangshuhe/A100/home/LLaMA-Factory/data/hermes_tag_ratio_0_2.json")
    # only_multi_turn_wild_chat_training = extract_multi_turn(data=wild_chat_training, num_=10000)
    # print(len(only_multi_turn_wild_chat_training+hermes_200k))
    # write_json_file(data=only_multi_turn_wild_chat_training+hermes_200k, output_file="/gpfs/wangshuhe/A100/home/LLaMA-Factory/data/hermes_2_5_0_2_10k_wildchat_gpt_4_multi_turn.json")

    # yi_single_turn = read_jsonl(input_file="/gpfs/public/align/wangshuhe/A800/open-instruct/data/processed/yi_data/yi_single_turn.jsonl")
    # yi_single_turn = transfer_training(data=yi_single_turn)
    # # print(len(yi_single_turn))
    # # write_json_file(data=yi_single_turn, output_file="/gpfs/wangshuhe/A100/home/LLaMA-Factory/data/yi_mixtures.json")
    # only_single_turn = extract_single_turn(data=yi_single_turn, num_=-1)
    # print(len(only_single_turn))
    # write_json_file(data=only_single_turn, output_file="/gpfs/wangshuhe/A100/home/LLaMA-Factory/data/yi_only_single_turn.json")

    # sampled_meta_math = read_jsonl(input_file="/gpfs/public/align/wangshuhe/A100/home/prompt2ppl/open-instruct/output_dir/metamath/min_max_sampled_0_1.jsonl")
    # sampled_meta_math = transfer_training(data=sampled_meta_math)
    # print(len(sampled_meta_math))
    # write_json_file(data=sampled_meta_math, output_file="/gpfs/public/align/wangshuhe/A100/home/LLaMA-Factory/data/min_max_sampled_0_1.json")

    # sampled_meta_math = read_jsonl(input_file="/gpfs/public/align/wangshuhe/A100/home/prompt2ppl/open-instruct/output_dir/metamath/min_max_sampled_0_1.jsonl")
    # sampled_meta_math = transfer_training(data=sampled_meta_math)
    # tag_hermes = read_jsonl(input_file="/gpfs/public/align/wangshuhe/A800/open-instruct/data/processed/OpenHermes_2_5/openhermes2_5_llama_3_70B_tag.jsonl")
    # specifci_tag = ["Linguistics and language", "History", "Computer sciences", "Education", "Literature"]
    # formatted_hermes = transfer_training_specific_tag(data=tag_hermes, specific_tag=specifci_tag)
    # print(len(sampled_meta_math+formatted_hermes))
    # write_json_file(data=sampled_meta_math+formatted_hermes, output_file="/gpfs/public/align/wangshuhe/A100/home/LLaMA-Factory/data/min_max_sampled_0_1_hermes_top_5.json")

    # sampled_hermes = read_jsonl(input_file="/gpfs/public/align/wangshuhe/A100/home/prompt2ppl/open-instruct/output_dir/hermes/openhermes2_5_sampled_0_2.jsonl")
    # sampled_hermes = transfer_training(data=sampled_hermes)
    # print(len(sampled_hermes))
    # write_json_file(data=sampled_hermes, output_file="/gpfs/public/align/wangshuhe/A100/home/LLaMA-Factory/data/hermes_min_max_sampled_0_2.json")

    sampled_hermes = read_jsonl(input_file="/gpfs/public/align/wangshuhe/A100/home/prompt2ppl/open-instruct/output_dir/hermes/openhermes2_5_equal_split_3.jsonl")
    sampled_hermes = transfer_training(data=sampled_hermes)
    print(len(sampled_hermes))
    write_json_file(data=sampled_hermes, output_file="/gpfs/public/align/wangshuhe/A100/home/LLaMA-Factory/data/hermes_equal_split_3.json")