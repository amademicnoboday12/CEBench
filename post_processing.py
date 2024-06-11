def post_processing_mental(response):
    import re
    score_search = re.search(r'Total score: (\d{1,2})', response)
    if score_search and 0 <= int(score_search.group(1)) <= 24:
        score = score_search.group(1)
    else:
        score = "-1"
    return score
