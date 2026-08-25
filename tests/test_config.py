from config import CONFIG


def test_default_config_contract():
    assert CONFIG.seed == 42
    assert CONFIG.image_size == 64
    assert CONFIG.num_classes == 10
    assert CONFIG.samples_per_class == 50
    assert (CONFIG.train_per_class, CONFIG.val_per_class, CONFIG.test_per_class) == (35, 7, 8)
    assert CONFIG.embedding_dim == 160
    assert CONFIG.batch_size == 32
    assert CONFIG.max_epochs == 80
    assert CONFIG.early_stopping_patience == 12
    assert CONFIG.learning_rate == 1e-3
    assert CONFIG.weight_decay == 1e-4
    assert CONFIG.device == "cuda:0"


def test_config_extra_defaults():
    assert CONFIG.min_faces_per_person == 20
    assert CONFIG.lfw_resize == 0.5
    assert CONFIG.dropout == 0.4
    assert CONFIG.horizontal_flip_probability == 0.5
    assert CONFIG.rotation_degrees == 8
    assert CONFIG.translation_fraction == 0.05
    assert CONFIG.scale_range == (0.95, 1.05)
