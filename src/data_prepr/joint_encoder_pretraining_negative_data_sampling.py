import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split


def negative_sampling(df, data_type=None, num_negative=5):

    print(f"********* Generating Pos Neg Data for : {data_type} *********")

    pos_data_list = df.values.tolist()

    df["label"] = int(1)

    # df.reset_index(drop=True, inplace=True)
    print(f"{data_type} datframe shape {df.shape}")
    print(df.head())
    print()

    unique_concepts = df["concept"].unique()
    unique_properties = df["property"].unique()

    print(f"Number of Unique Concepts in Dataframe :", len(unique_concepts), flush=True)

    all_negative_data = []

    for concept in unique_concepts:

        concept_data = df[df["concept"] == concept]
        properties_for_concept = concept_data["property"].unique()
        conjuct_properties_for_concept = concept_data["conjuct_prop"].unique()

        num_record = len(concept_data)

        print()
        print(f"Generating Negative Data for Concept : {concept}", flush=True)
        print(f"Positive data for concept in DF : {concept_data.shape}", flush=True)

        print("Data For concept", flush=True)
        print(concept_data, flush=True)
        print(f"Properties for Concept", flush=True)
        print(properties_for_concept, flush=True)
        print(f"Conjuction Properties for Concept", flush=True)
        print(conjuct_properties_for_concept, flush=True)

        total_neg_num = num_record * num_negative

        print(f"Total Number of Negative Records to be generated : {total_neg_num}")

        rest_df = df[df["concept"] != concept]
        print(f"Rest DF shape after removing concept : {rest_df.shape}")
        rest_df = rest_df[~rest_df["property"].isin(properties_for_concept)]

        print(
            f"Rest DF shape after removing concepts's properties : {rest_df.shape}",
            flush=True,
        )

        concept_neg_data = []

        while True:

            concept = concept.strip()
            neg_properties = list(rest_df["property"].sample(n=total_neg_num))
            conjuct_props = random.choices(
                conjuct_properties_for_concept, k=total_neg_num
            )

            neg_data = [
                [concept, conjuct_prop, neg_prop]
                for conjuct_prop, neg_prop in zip(conjuct_props, neg_properties)
            ]
            print(f"neg_data length :", len(neg_data), flush=True)

            if len(concept_neg_data) < total_neg_num:
                for x in neg_data:
                    if not (x in pos_data_list):
                        if not (x in all_negative_data):

                            all_negative_data.append(x)
                            concept_neg_data.append(x)

                            if len(concept_neg_data) == total_neg_num:
                                break

            if len(concept_neg_data) == total_neg_num:
                break

        print(
            f"Number of negative records generated : {len(concept_neg_data)}",
            flush=True,
        )
        print(f"Negative Records", flush=True)
        print(concept_neg_data, flush=True)
        print()

    _ = [x.insert(len(x), int(0)) for x in all_negative_data]

    print("all_negative_data")
    print(all_negative_data)

    all_neg_data_df = pd.DataFrame.from_records(
        all_negative_data, columns=["concept", "conjuct_prop", "property", "label"]
    )

    neg_data_duplicate_records = all_neg_data_df[
        all_neg_data_df.duplicated(["concept", "property"])
    ]

    print()
    print(f"all_neg_data_df.shape : {all_neg_data_df.shape}", flush=True)
    print(
        f"neg_data_duplicate_records.shape : {neg_data_duplicate_records.shape}",
        flush=True,
    )
    print()

    print(f"Checking overlap between positive and negative data", flush=True)
    pos_neg_overlap_df = df.merge(
        all_neg_data_df, how="inner", on=["concept", "property"], indicator=False
    )
    print(f"Positive and Negative Overlapped Dataframe", flush=True)
    print(pos_neg_overlap_df, flush=True)
    print()

    pos_neg_df = pd.concat([df, all_neg_data_df], axis=0, ignore_index=True)

    print("DF after adding negative data", flush=True)
    print(pos_neg_df.shape, flush=True)

    duplicate_records = pos_neg_df[pos_neg_df.duplicated(["concept", "property"])]

    print(f"Duplicate Records : {duplicate_records.shape}", flush=True)
    print(
        f"Duplicate record label value count: {duplicate_records['label'].value_counts()}",
        flush=True,
    )
    print()

    pos_neg_df = pos_neg_df[
        ~pos_neg_df.duplicated(subset=["concept", "property"], keep="first")
    ]

    pos_neg_df.drop_duplicates(inplace=True)
    pos_neg_df.dropna(how="any", inplace=True)

    pos_neg_df.dropna(axis=0, subset=["concept"], inplace=True)
    pos_neg_df.dropna(axis=0, subset=["property"], inplace=True)
    pos_neg_df.dropna(axis=0, subset=["label"], inplace=True)

    pos_neg_df = pos_neg_df.sample(frac=1)

    print(f"Dataframe after removing duplicates : {pos_neg_df.shape}", flush=True)

    save_path = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data"

    if data_type == "train":

        file_name = os.path.join(
            save_path,
            f"{num_negative}_neg_train_random_and_similar_conjuct_properties.tsv",
        )
        pos_neg_df.to_csv(file_name, sep="\t", index=None, header=None)

    elif data_type == "valid":

        file_name = os.path.join(
            save_path,
            f"{num_negative}_neg_valid_random_and_similar_conjuct_properties.tsv",
        )
        pos_neg_df.to_csv(file_name, sep="\t", index=None, header=None)

    return pos_neg_df


def comma_and_space(text):

    text = text.strip().split(",")
    text = ", ".join(text)

    return text


local_file_name = "siamese_concept_property/data/train_data/joint_encoder_property_conjuction_data/random_and_similar_conjuct_properties.tsv"
hawk_file_name = "/scratch/c.scmag3/biencoder_concept_property/data/train_data/joint_encoder_property_conjuction_data/random_and_similar_conjuct_properties.tsv"


df_rand_sim = pd.read_csv(
    hawk_file_name, sep="\t", names=["concept", "conjuct_prop", "property"]
)

df_rand_sim["conjuct_prop"] = df_rand_sim["conjuct_prop"].apply(comma_and_space)

print(df_rand_sim)

unique_concepts = df_rand_sim["concept"].unique()

print(f"Unique Concepts : {len(unique_concepts)}", flush=True)

# valid_concepts = np.random.choice(a=unique_concepts, size = int(0.1 * len(unique_concepts)), replace=False)

valid_concepts = np.random.choice(a=unique_concepts, size=1000, replace=False)

print(f"Number of test concepts : {len(valid_concepts)}", flush=True)

valid_df = df_rand_sim[df_rand_sim["concept"].isin(valid_concepts)]
train_df = df_rand_sim[~df_rand_sim["concept"].isin(valid_concepts)]

print()
print("Total Random and Similar DF shape :", df_rand_sim.shape, flush=True)
print("Train DF shape :", train_df.shape, train_df.columns, flush=True)
print("Test DF shape :", valid_df.shape, valid_df.columns, flush=True)

print("Checking Train Test DF Merge", flush=True)
df1 = train_df.merge(valid_df, how="inner", on=["concept"], indicator=False)

print(df1)
assert df1.empty, "Error: Train Test Concepts Overlap"


print(f"************ Generating Negative Train Data ************", flush=True)

pos_neg_train_df = negative_sampling(train_df, "train", num_negative=5)

print(f"************ Generating Negative Valid Data ************", flush=True)
pos_neg_valid_df = negative_sampling(valid_df, "valid", num_negative=5)

print(f"************ Negative Data Generation Process Ends ************", flush=True)

