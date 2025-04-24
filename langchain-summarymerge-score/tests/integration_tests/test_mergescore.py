from langchain_summarymerge_score import SummaryMergeScoreTool

def test_summary_merge_score_tool_endpoint():
    summary_merger = SummaryMergeScoreTool(
        api_base="http://localhost:8000/merge_summaries"
        )

    summaries = {
            "chunk_0": "Start time: 0 End time: 30\n**Overall Summary**\nThe video captures a sequence of moments inside a "
                    "retail store, focusing on the checkout area and the surrounding aisles. The timestamp indicates the "
                    "footage was taken on Tuesday, May 21, 2024, at 06:42:42.\n\n**Activity Observed**\n1. The video shows "
                    "a relatively empty store with no visible customers at the checkout counter.\n2. The shelves on the "
                    "right side are stocked with various products, and the floor is clean and clear of obstructions.\n3. "
                    "There is a green waste bin placed near the checkout counter.\n4. The store appears well-lit, "
                    "and the checkout area is equipped with modern electronic devices.\n\n**Potential Suspicious "
                    "Activity**\n1. There is no visible evidence of shoplifting or any suspicious behavior in the provided "
                    "frames. The store appears orderly, and there are no signs of tampering or "
                    "theft.\n\n**Conclusion**\nBased on the analysis, the video shows a typical scene in a retail store "
                    "with no immediate signs of shoplifting or suspicious activity. The store is clean, organized, "
                    "and operational without any disturbances.",
            "chunk_1": "Start time: 28 End time: 30.002969\n**Overall Summary**\nThe video captures a sequence of moments "
                    "inside a retail store, focusing on the checkout area and the surrounding aisles. The timestamp "
                    "indicates the footage was taken on Tuesday, May 21, 2024, at 06:42:52.\n\n**Activity Observed**\n1. "
                    "The video shows a cashier's station with a computer monitor and a cash drawer.\n2. The aisles are "
                    "stocked with various products, including snacks and beverages.\n3. There is a visible customer "
                    "interaction area near the checkout counter.\n4. The floor is clean and well-maintained.\n5. The store "
                    "appears to be open and operational during the time the video was recorded.\n\n**Potential Suspicious "
                    "Activity**\n1. No overt signs of shoplifting or suspicious behavior are observed in the provided "
                    "frames. The cashier and the customer interaction area remain empty throughout the "
                    "sequence.\n\n**Conclusion**\nBased on the analysis, there is no evidence of shoplifting or suspicious "
                    "activity in the provided video frames. The store appears to be functioning normally without any "
                    "immediate concerns."
            }

    output = summary_merger.invoke({"summaries": summaries})
    print(output)
    assert isinstance(output, dict)
    assert "overall_summary" in output
    assert "anomaly_score" in output

def test_summary_merge_score_tool_model_id():
    summary_merger = SummaryMergeScoreTool(
        model_id="llmware/llama-3.2-3b-instruct-ov",
        device="GPU")

    summaries = {
            "chunk_0": "Start time: 0 End time: 30\n**Overall Summary**\nThe video captures a sequence of moments inside a "
                    "retail store, focusing on the checkout area and the surrounding aisles. The timestamp indicates the "
                    "footage was taken on Tuesday, May 21, 2024, at 06:42:42.\n\n**Activity Observed**\n1. The video shows "
                    "a relatively empty store with no visible customers at the checkout counter.\n2. The shelves on the "
                    "right side are stocked with various products, and the floor is clean and clear of obstructions.\n3. "
                    "There is a green waste bin placed near the checkout counter.\n4. The store appears well-lit, "
                    "and the checkout area is equipped with modern electronic devices.\n\n**Potential Suspicious "
                    "Activity**\n1. There is no visible evidence of shoplifting or any suspicious behavior in the provided "
                    "frames. The store appears orderly, and there are no signs of tampering or "
                    "theft.\n\n**Conclusion**\nBased on the analysis, the video shows a typical scene in a retail store "
                    "with no immediate signs of shoplifting or suspicious activity. The store is clean, organized, "
                    "and operational without any disturbances.",
            "chunk_1": "Start time: 28 End time: 30.002969\n**Overall Summary**\nThe video captures a sequence of moments "
                    "inside a retail store, focusing on the checkout area and the surrounding aisles. The timestamp "
                    "indicates the footage was taken on Tuesday, May 21, 2024, at 06:42:52.\n\n**Activity Observed**\n1. "
                    "The video shows a cashier's station with a computer monitor and a cash drawer.\n2. The aisles are "
                    "stocked with various products, including snacks and beverages.\n3. There is a visible customer "
                    "interaction area near the checkout counter.\n4. The floor is clean and well-maintained.\n5. The store "
                    "appears to be open and operational during the time the video was recorded.\n\n**Potential Suspicious "
                    "Activity**\n1. No overt signs of shoplifting or suspicious behavior are observed in the provided "
                    "frames. The cashier and the customer interaction area remain empty throughout the "
                    "sequence.\n\n**Conclusion**\nBased on the analysis, there is no evidence of shoplifting or suspicious "
                    "activity in the provided video frames. The store appears to be functioning normally without any "
                    "immediate concerns."
            }

    output = summary_merger.invoke({"summaries": summaries})
    print(output)
    assert isinstance(output, dict)
    assert "overall_summary" in output
    assert "anomaly_score" in output

if __name__ == "__main__":
    test_summary_merge_score_tool_endpoint()
    test_summary_merge_score_tool_model_id()
    print("All tests passed!")