"""Tests for word grouping feature."""

import pytest
import numpy as np
from wordcloud import WordCloud
from wordcloud.group_placement import GroupRegion, GroupAwarePlacer


def test_group_region_update():
    region = GroupRegion()
    region.update((10, 10), (20, 30))
    
    assert region.centroid == (20.0, 25.0)
    assert len(region.word_positions) == 1
    assert region.total_area == 600
    
    region.update((50, 50), (30, 40))
    assert len(region.word_positions) == 2


def test_group_aware_placer_init():
    placer = GroupAwarePlacer(mode='probabilistic', radius=50, strength=0.7)
    assert placer.mode == 'probabilistic'
    assert placer.radius == 50
    assert placer.strength == 0.7


def test_grouping_func_validation():
    with pytest.raises(TypeError, match="grouping_func must be callable"):
        WordCloud(grouping_func="not a function")
    
    with pytest.raises(ValueError, match="group_proximity must be"):
        WordCloud(group_proximity='invalid')
    
    with pytest.raises(ValueError, match="group_radius must be non-negative"):
        WordCloud(group_radius=-10)
    
    with pytest.raises(ValueError, match="group_strength must be between"):
        WordCloud(group_strength=1.5)
    
    with pytest.raises(ValueError, match="group_strength must be between"):
        WordCloud(group_strength=-0.1)


def test_wordcloud_with_probabilistic_grouping():
    text = "python code programming software python code"
    
    def simple_grouper(word):
        return 'tech' if word in ['python', 'code', 'programming', 'software'] else None
    
    wc = WordCloud(
        grouping_func=simple_grouper,
        group_proximity='probabilistic',
        group_strength=0.8,
        max_words=10,
        random_state=42,
        collocations=False
    )
    
    wc.generate(text)
    assert len(wc.layout_) > 0


def test_wordcloud_with_strict_grouping():
    text = "apple apricot avocado banana berry"
    
    def simple_grouper(word):
        return 'group_a' if word in ['apple', 'apricot', 'avocado'] else 'group_b'
    
    wc = WordCloud(
        grouping_func=simple_grouper,
        group_proximity='strict',
        group_radius=100,
        max_words=10,
        random_state=42,
        collocations=False
    )
    
    wc.generate(text)
    assert len(wc.layout_) > 0


def test_backward_compatibility_no_grouping():
    text = "hello world test python code"
    
    wc_old = WordCloud(max_words=10, random_state=42, collocations=False)
    wc_new = WordCloud(max_words=10, random_state=42, collocations=False, grouping_func=None)
    
    wc_old.generate(text)
    wc_new.generate(text)
    
    assert len(wc_old.layout_) > 0
    assert len(wc_new.layout_) > 0


def test_none_grouping_func():
    text = "hello world test python code"
    
    wc = WordCloud(
        grouping_func=lambda w: None,
        max_words=10,
        random_state=42,
        collocations=False
    )
    
    wc.generate(text)
    assert len(wc.layout_) > 0


def test_grouping_with_mask():
    text = "python code programming software python code"
    mask = np.zeros((200, 400), dtype=int)
    mask[50:150, 100:300] = 255
    
    def simple_grouper(word):
        return 'tech' if word in ['python', 'code', 'programming', 'software'] else None
    
    wc = WordCloud(
        grouping_func=simple_grouper,
        group_proximity='probabilistic',
        mask=mask,
        max_words=20,
        random_state=42,
        collocations=False
    )
    
    wc.generate(text)
    assert len(wc.layout_) > 0


def test_multiple_groups():
    text = """
    python code programming software development
    tree forest river mountain nature
    happy joy love emotion feeling
    """ * 3
    
    def semantic_grouper(word):
        categories = {
            'tech': ['python', 'code', 'programming', 'software', 'development'],
            'nature': ['tree', 'forest', 'river', 'mountain', 'nature'],
            'emotion': ['happy', 'joy', 'love', 'emotion', 'feeling']
        }
        for group, words in categories.items():
            if word in words:
                return group
        return None
    
    wc = WordCloud(
        grouping_func=semantic_grouper,
        group_proximity='probabilistic',
        group_strength=0.9,
        max_words=30,
        random_state=42,
        collocations=False
    )
    
    wc.generate(text)
    assert len(wc.layout_) > 0
