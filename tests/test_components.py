#!/usr/bin/env python3
"""
Component Tests — Verify each module works independently.

Run: python -m pytest tests/test_components.py -v
  or: python tests/test_components.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import tempfile


def test_rssm_world_model():
    """Test the RSSM world model forward pass and training."""
    from developmental_ai.world_model.rssm import WorldModel

    obs_dim, action_dim = 4, 2
    model = WorldModel(
        obs_dim=obs_dim, action_dim=action_dim,
        stochastic_size=8, stochastic_classes=8,
        deterministic_size=64, hidden_dim=32,
    )

    # Test observe sequence
    batch, seq_len = 4, 10
    obs = torch.randn(batch, seq_len, obs_dim)
    actions = torch.randn(batch, seq_len, action_dim)
    rewards = torch.randn(batch, seq_len)
    continues = torch.ones(batch, seq_len)

    states, infos = model.observe_sequence(obs, actions)
    assert states["h"].shape == (batch, seq_len, 64), f"h shape: {states['h'].shape}"
    assert states["z"].shape == (batch, seq_len, 64), f"z shape: {states['z'].shape}"

    # Test training step
    losses = model.train_step(obs, actions, rewards, continues)
    assert "total" in losses
    assert losses["total"] > 0
    print(f"  RSSM losses: {losses}")

    # Test imagination
    initial_state = model.rssm.initial_state(2, torch.device("cpu"))
    imagined = model.imagine_trajectory(
        initial_state,
        policy_fn=lambda x: torch.randn(2, action_dim),
        horizon=5,
    )
    assert imagined["latents"].shape[1] == 5
    assert imagined["rewards"].shape == (2, 5, 1)

    print("  [PASS] RSSM World Model")


def test_replay_buffer():
    """Test replay buffer storage and sampling."""
    from developmental_ai.world_model.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=1000, obs_dim=4, action_dim=2)

    # Fill with fake data (simulate a few episodes)
    for ep in range(5):
        for step in range(50):
            obs = np.random.randn(4).astype(np.float32)
            action = np.random.randint(2)
            reward = float(np.random.randn())
            done = (step == 49)
            buffer.add(obs, action, reward, done)

    assert len(buffer) == 250

    # Sample sequences
    batch = buffer.sample_sequences(batch_size=4, seq_len=10)
    assert batch["observations"].shape == (4, 10, 4)
    assert batch["actions"].shape == (4, 10, 2)
    assert batch["rewards"].shape == (4, 10)

    print("  [PASS] Replay Buffer")


def test_icm_curiosity():
    """Test the Intrinsic Curiosity Module."""
    from developmental_ai.curiosity.icm import IntrinsicCuriosityModule

    obs_dim, action_dim = 4, 2
    icm = IntrinsicCuriosityModule(
        obs_dim=obs_dim, action_dim=action_dim,
        feature_dim=32, hidden_dim=32,
    )

    batch = 8
    obs = torch.randn(batch, obs_dim)
    action = torch.randn(batch, action_dim)
    next_obs = torch.randn(batch, obs_dim)

    # Test intrinsic reward computation
    reward = icm.compute_intrinsic_reward(obs, action, next_obs)
    assert reward.shape == (batch,), f"Reward shape: {reward.shape}"

    # Test training step
    metrics = icm.train_step(obs, action, next_obs)
    assert "icm_total" in metrics
    assert "icm_forward" in metrics
    assert "icm_inverse" in metrics
    print(f"  ICM metrics: {metrics}")

    # Test novelty score
    score = icm.get_novelty_score(obs[:1])
    assert isinstance(score, float)

    # Test exploration ratio decay
    icm.update_exploration_ratio(skill_count=5)
    assert icm.exploration_ratio < 1.0

    print("  [PASS] ICM Curiosity")


def test_knowledge_graph():
    """Test the in-memory knowledge graph."""
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph, SymbolicFact, ActionRule, FactExtractor
    )

    kg = InMemoryKnowledgeGraph()

    # Add facts
    f1 = SymbolicFact("cart", "has_velocity", "positive", confidence=0.9)
    f2 = SymbolicFact("push_right", "causes", "cart_moves_right", confidence=0.8)
    f3 = SymbolicFact("pole", "has_angle", "small", confidence=0.9)

    assert kg.add_fact(f1) == True   # Novel
    assert kg.add_fact(f1) == False  # Duplicate
    kg.add_fact(f2)
    kg.add_fact(f3)

    # Query
    results = kg.query(subject="cart")
    assert len(results) == 1
    results = kg.query(relation="causes")
    assert len(results) == 1

    # Add action rule
    rule = ActionRule(
        preconditions={"pole_angle": "small"},
        action="push_right",
        effects={"cart_velocity": "positive"},
        confidence=0.7,
    )
    kg.add_action_rule(rule)

    # Get applicable rules
    applicable = kg.get_applicable_rules({"pole_angle": "small"})
    assert len(applicable) == 1

    stats = kg.get_stats()
    assert stats["num_facts"] == 3
    assert stats["num_action_rules"] == 1

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    kg.save(path)

    kg2 = InMemoryKnowledgeGraph()
    kg2.load(path)
    assert kg2.get_stats()["num_facts"] == 3
    os.unlink(path)

    # Test fact extractor
    extractor = FactExtractor(
        obs_labels=["cart_pos", "cart_vel", "pole_angle", "pole_vel"],
    )
    obs = np.array([0.1, 0.5, -0.2, 0.3])
    next_obs = np.array([0.2, 0.6, -0.1, 0.4])
    facts, rule = extractor.extract_transition_facts(obs, 1, next_obs, 1.0)
    assert len(facts) > 0

    print("  [PASS] Knowledge Graph")


def test_skill_bank():
    """Test skill bank save and retrieve."""
    from developmental_ai.skill_bank.skill_bank import SkillBank, MasteryDetector

    with tempfile.TemporaryDirectory() as tmpdir:
        bank = SkillBank(storage_dir=tmpdir)

        # Save a skill
        policy_state = {"layer1.weight": torch.randn(32, 4)}
        skill = bank.save_skill(
            skill_id="test_skill_001",
            name="Test Balance",
            policy_state_dict=policy_state,
            success_rate=0.9,
            total_episodes=100,
            avg_reward=150.0,
            context_embedding=np.random.randn(16),
        )

        assert skill.is_mastered
        assert bank.get_skill_count() == 1

        # Retrieve skills
        # min_similarity=0.0 explicitly: this case tests the save/retrieve
        # ROUND TRIP, not the relevance policy. Both embeddings are random
        # 16-d gaussians, so their normalized similarity sits at ~0.5 — the
        # configured default cut — and the assertion below would be a coin
        # flip rather than a test.
        retrieved = bank.retrieve_skills(
            context_embedding=np.random.randn(16),
            max_skills=3,
            min_similarity=0.0,
        )
        assert len(retrieved) == 1

        # Load policy
        loaded = bank.load_skill_policy("test_skill_001")
        assert loaded is not None
        assert "layer1.weight" in loaded

        # Test mastery detector
        detector = MasteryDetector(mastery_threshold=0.8, mastery_window=20, min_episodes=10)
        for _ in range(15):
            detector.record_episode(reward=100.0, success=True)
        is_mastered, level = detector.check_mastery()
        assert is_mastered
        assert level >= 1.0

    print("  [PASS] Skill Bank")


def test_policy():
    """Test the standalone actor-critic policy."""
    from developmental_ai.policy.actor_critic import StandaloneActorCritic, RewardMixer

    obs_dim, action_dim = 4, 2
    policy = StandaloneActorCritic(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=32, continuous=False,
    )

    # Test action selection
    obs = np.random.randn(obs_dim).astype(np.float32)
    action, info = policy.select_action(obs)
    assert isinstance(action, int)
    assert "log_prob" in info
    assert "value" in info

    # Collect a rollout and train
    for _ in range(50):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action, info = policy.select_action(obs)
        policy.store_transition(obs, action, 1.0, False, info["log_prob"], info["value"])

    metrics = policy.train_step(n_epochs=3)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    print(f"  Policy metrics: {metrics}")

    # Test state dict save/load
    state = policy.get_state_dict()
    assert "actor" in state
    assert "critic" in state

    # Test reward mixer
    mixer = RewardMixer(intrinsic_weight=0.7, extrinsic_weight=0.3)
    mixed = mixer.mix(1.0, 2.0)
    assert abs(mixed - 1.3) < 0.01
    mixer.decay()
    assert mixer.intrinsic_weight < 0.7

    print("  [PASS] Policy")


def test_environment_wrapper():
    """Test the environment wrapper."""
    from developmental_ai.environments.wrappers import make_env, get_env_labels

    env, curriculum = make_env("CartPole-v1", normalize=True, curriculum=True)

    assert env.obs_dim == 4
    assert env.action_dim == 2
    assert env.is_discrete

    # Test reset and step
    obs, info = env.reset()
    assert obs.shape == (4,)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (4,)
        assert "extrinsic_reward" in info
        if terminated or truncated:
            obs, info = env.reset()

    # Test labels
    obs_labels, action_labels = get_env_labels("CartPole-v1")
    assert obs_labels == ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]
    assert action_labels == ["push_left", "push_right"]

    env.close()
    print("  [PASS] Environment Wrapper")


def test_graph_embeddings():
    """Test graph embedding trainer (fallback mode)."""
    from developmental_ai.knowledge_graph.graph_embeddings import (
        KGEmbeddingTrainer, KnowledgeGraphGNN, SymbolicNeuralGate
    )

    # Test embedding trainer
    trainer = KGEmbeddingTrainer(embedding_dim=16)
    triples = [
        ("cart", "has_velocity", "positive"),
        ("push_right", "causes", "movement"),
        ("pole", "has_angle", "small"),
        ("action_push", "leads_to", "reward"),
    ]
    result = trainer.train(triples)
    assert "num_entities" in result

    emb = trainer.get_entity_embedding("cart")
    assert emb is not None
    assert emb.shape == (16,)

    # Test GNN
    gnn = KnowledgeGraphGNN(input_dim=16, hidden_dim=32, output_dim=16)
    node_features = torch.randn(5, 16)
    knowledge_vec = gnn(node_features)
    assert knowledge_vec.shape == (16,)

    # Test symbolic-neural gate
    gate = SymbolicNeuralGate(neural_dim=64, symbolic_dim=16, output_dim=32)
    neural = torch.randn(1, 64)
    symbolic = torch.randn(1, 16)
    blended = gate(neural, symbolic)
    assert blended.shape == (1, 32)

    print("  [PASS] Graph Embeddings")


def test_symbolic_decoder():
    """Test the symbolic decoder (latent → discrete categories)."""
    from developmental_ai.world_model.symbolic_decoder import (
        SymbolicDecoderManager, ObservationDiscretizer, SymbolicDecoder
    )

    obs_dim = 4
    latent_dim = 128  # Small for testing
    num_categories = 5

    # Test ObservationDiscretizer
    disc = ObservationDiscretizer(
        obs_dim=obs_dim, num_categories=num_categories,
        obs_labels=["cart_pos", "cart_vel", "pole_angle", "pole_vel"],
    )
    # Feed some observations to build statistics
    for _ in range(100):
        obs = np.random.randn(obs_dim).astype(np.float32)
        disc.update(obs)

    indices = disc.discretize(np.array([0.0, 0.0, 0.0, 0.0]))
    assert indices.shape == (obs_dim,)
    assert all(0 <= idx < num_categories for idx in indices)

    dim_name, cat_name = disc.index_to_label(0, 2)
    assert dim_name == "cart_pos"
    assert cat_name == "medium"

    # Test SymbolicDecoder network
    decoder = SymbolicDecoder(latent_dim=latent_dim, obs_dim=obs_dim, num_categories=num_categories)
    latent = torch.randn(8, latent_dim)

    logits = decoder(latent)
    assert logits.shape == (8, obs_dim, num_categories), f"Logits shape: {logits.shape}"

    cats, confs = decoder.predict_categories(latent)
    assert cats.shape == (8, obs_dim)
    assert confs.shape == (8, obs_dim)
    assert (confs >= 0).all() and (confs <= 1).all(), "Confidences must be in [0, 1]"

    # Test loss computation
    targets = torch.randint(0, num_categories, (8, obs_dim))
    loss = decoder.compute_loss(latent, targets)
    assert loss.item() > 0
    print(f"  Decoder loss: {loss.item():.4f}")

    # Test full SymbolicDecoderManager
    manager = SymbolicDecoderManager(
        latent_dim=latent_dim, obs_dim=obs_dim, num_categories=num_categories,
        hidden_dim=64, confidence_threshold=0.3,
        obs_labels=["cart_pos", "cart_vel", "pole_angle", "pole_vel"],
    )

    # Feed observations to build discretizer stats
    for _ in range(50):
        manager.update_discretizer(np.random.randn(obs_dim).astype(np.float32))

    # Train the symbolic decoder
    latent_batch = torch.randn(16, 10, latent_dim)  # (batch, seq, latent)
    obs_batch = torch.randn(16, 10, obs_dim)
    metrics = manager.train_step(latent_batch, obs_batch)
    assert "symbolic_decoder_loss" in metrics
    assert "symbolic_decoder_accuracy" in metrics
    assert "symbolic_decoder_confidence" in metrics
    print(f"  Manager train: loss={metrics['symbolic_decoder_loss']:.4f}, "
          f"acc={metrics['symbolic_decoder_accuracy']:.2%}, "
          f"conf={metrics['symbolic_decoder_confidence']:.3f}")

    # Train a few more steps so accuracy improves
    for _ in range(20):
        manager.train_step(latent_batch, obs_batch)

    # Extract facts from a latent state
    single_latent = torch.randn(latent_dim)
    facts = manager.extract_facts(single_latent, timestep=42)
    assert isinstance(facts, list)
    # With low confidence threshold, we should get some facts
    print(f"  Extracted {len(facts)} facts from latent state")
    for f in facts[:3]:
        print(f"    {f.subject} --[{f.relation}]--> {f.obj} (conf={f.confidence:.2f}, src={f.source})")

    # Verify fact source is marked correctly
    for f in facts:
        assert f.source == "symbolic_decoder"

    # Check stats
    stats = manager.stats
    assert stats["symbolic_decoder_train_steps"] > 0
    assert stats["discretizer_obs_count"] > 0

    print("  [PASS] Symbolic Decoder")


def test_knowledge_integrator():
    """Test the KnowledgeIntegrator: GNN + Gate → augmented latents."""
    from developmental_ai.core.glue_layer import KnowledgeIntegrator
    from developmental_ai.knowledge_graph.graph_embeddings import KGEmbeddingTrainer
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph, SymbolicFact,
    )

    embedding_dim = 16
    latent_dim = 128

    integrator = KnowledgeIntegrator(
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        gnn_hidden_dim=32,
        gnn_output_dim=16,
    )

    # Without knowledge, augment_latent is identity
    latent = torch.randn(4, latent_dim)
    augmented = integrator.augment_latent(latent)
    assert torch.allclose(augmented, latent), "Should pass through when no knowledge"
    assert not integrator.has_knowledge

    # Build a small KG and train embeddings
    kg = InMemoryKnowledgeGraph()
    kg.add_fact(SymbolicFact("cart", "has_velocity", "positive"))
    kg.add_fact(SymbolicFact("push_right", "causes", "movement"))
    kg.add_fact(SymbolicFact("pole", "has_angle", "small"))
    kg.add_fact(SymbolicFact("action_push", "leads_to", "reward"))

    embedder = KGEmbeddingTrainer(embedding_dim=embedding_dim)
    embedder.train(kg.get_triples_for_embedding())

    # Update knowledge vector
    kv = integrator.update_knowledge_vector(embedder, kg)
    assert kv is not None
    assert kv.shape == (16,), f"Knowledge vector shape: {kv.shape}"
    assert integrator.has_knowledge

    # Now augment_latent should blend knowledge in
    augmented = integrator.augment_latent(latent)
    assert augmented.shape == (4, latent_dim)
    # Gate initialized to favor neural, so augmented should be close but not identical
    diff = (augmented - latent).abs().mean().item()
    assert diff > 0, "Augmented should differ from raw latent"
    print(f"  Augmented latent diff from raw: {diff:.4f}")

    print("  [PASS] Knowledge Integrator")


def test_glue_layer():
    """Test the full glue layer: goals, mastery, curriculum, skill selection."""
    from developmental_ai.core.glue_layer import (
        GlueLayer, GoalGenerator, AdvancedMasteryDetector,
        AdvancedCurriculumManager, SkillSelector,
    )
    from developmental_ai.knowledge_graph.knowledge_graph import (
        InMemoryKnowledgeGraph, SymbolicFact, ActionRule,
    )
    from developmental_ai.skill_bank.skill_bank import SkillBank

    # ---- GoalGenerator ----
    kg = InMemoryKnowledgeGraph()
    gen = GoalGenerator(obs_dim=4, embedding_dim=16)

    # Empty KG → random goal
    goal = gen.generate_goal(kg)
    assert goal.shape == (16,)
    assert abs(np.linalg.norm(goal) - 1.0) < 0.01, "Goal should be unit norm"

    # Add some facts and generate again
    kg.add_fact(SymbolicFact("cart_pos", "has_value", "high"))
    kg.add_fact(SymbolicFact("cart_vel", "has_value", "low"))
    goal2 = gen.generate_goal(kg)
    assert goal2.shape == (16,)
    assert len(gen.goal_history) == 2
    print(f"  Goals generated: {len(gen.goal_history)}")

    # Scene-aware goal generation
    scene1 = {"entities": {"cart_pos": {"symbolic_value": "low"}, "pole_angle": {"symbolic_value": "medium"}}}
    scene2 = {"entities": {"cart_pos": {"symbolic_value": "high"}, "pole_angle": {"symbolic_value": "medium"}}}
    kg.save_scene_snapshot(scene1)
    kg.save_scene_snapshot(scene2)
    goal3 = gen.generate_goal(kg, scene_history=kg.scene_history)
    assert goal3.shape == (16,)

    # ---- AdvancedMasteryDetector ----
    detector = AdvancedMasteryDetector(min_samples=10)

    # Not enough data → not mastered
    is_mastered, signals = detector.detect_mastery(kg)
    assert not is_mastered
    assert signals["combined_score"] == 0.0

    # Feed declining KL and stable rewards
    for i in range(30):
        detector.record_kl(5.0 - i * 0.1)
        detector.record_reward(100.0 + np.random.randn() * 5)

    # Add high-confidence rules to KG
    for _ in range(5):
        kg.add_action_rule(ActionRule(
            preconditions={"pole_angle": "small"},
            action="push_right",
            effects={"cart_velocity": "positive"},
            confidence=0.9,
        ))

    is_mastered, signals = detector.detect_mastery(kg)
    assert signals["kl_trend"] > 0, f"KL signal should be positive: {signals}"
    assert signals["reward_stability"] > 0, f"Reward signal should be positive: {signals}"
    assert signals["rule_confidence"] > 0, f"Rule signal should be positive: {signals}"
    print(f"  Mastery signals: kl={signals['kl_trend']:.2f}, "
          f"reward={signals['reward_stability']:.2f}, "
          f"rules={signals['rule_confidence']:.2f}, "
          f"votes={signals['votes']:.0f}")

    # ---- AdvancedCurriculumManager ----
    with tempfile.TemporaryDirectory() as tmpdir:
        bank = SkillBank(storage_dir=tmpdir)
        curriculum = AdvancedCurriculumManager(max_difficulty=5, density_threshold=1.0)

        # Feed prediction errors that plateau
        for i in range(30):
            curriculum.record_prediction_error(1.0 - i * 0.01)

        # Add a skill
        bank.save_skill(
            skill_id="skill_001", name="Balance",
            policy_state_dict={"w": torch.randn(4)},
            success_rate=0.9, total_episodes=50, avg_reward=200.0,
        )

        recommended, curr_signals = curriculum.recommend_difficulty(bank, kg, 0)
        assert isinstance(recommended, int)
        assert "skill_tree_depth" in curr_signals
        assert "kg_density" in curr_signals
        assert "pred_error_slope" in curr_signals
        print(f"  Curriculum: recommended={recommended}, "
              f"depth={curr_signals['skill_tree_depth']:.0f}, "
              f"density={curr_signals['kg_density']:.1f}, "
              f"slope={curr_signals['pred_error_slope']:.4f}")

        # ---- SkillSelector ----
        selector = SkillSelector()

        # Add a skill with goal facts matching KG
        bank.save_skill(
            skill_id="skill_002", name="Push Right",
            policy_state_dict={"w": torch.randn(4)},
            success_rate=0.95, total_episodes=100, avg_reward=250.0,
            context_embedding=np.random.randn(16),
            goal_facts=[{"subject": "cart_pos", "relation": "has_value"}],
        )

        results = selector.select_skills(
            goal_embedding=np.random.randn(16),
            skill_bank=bank,
            knowledge_graph=kg,
            max_skills=3,
        )
        assert len(results) > 0
        skill, score = results[0]
        assert 0 <= score <= 1
        print(f"  Skill selection: {len(results)} skills, "
              f"best='{skill.name}' (score={score:.2f})")

        # Prerequisite filtering: add a skill with unmet prereqs
        bank.save_skill(
            skill_id="skill_003", name="Complex Behavior",
            policy_state_dict={"w": torch.randn(4)},
            success_rate=0.9, total_episodes=50, avg_reward=300.0,
            prerequisites=["skill_nonexistent"],
        )
        results_filtered = selector.select_skills(
            goal_embedding=np.random.randn(16),
            skill_bank=bank,
            knowledge_graph=kg,
            max_skills=5,
        )
        skill_ids = [s.skill_id for s, _ in results_filtered]
        assert "skill_003" not in skill_ids, "Skill with unmet prereqs should be filtered"

    # ---- Full GlueLayer integration ----
    glue = GlueLayer(
        latent_dim=128, obs_dim=4,
        embedding_dim=16, gnn_hidden_dim=32, gnn_output_dim=16,
        goal_embedding_dim=16,
    )

    # Record signals and check stats
    for i in range(25):
        glue.record_training_signals(kl_value=3.0 - i * 0.1, episode_reward=50.0)
        glue.record_training_signals(prediction_error=0.5 - i * 0.01)

    stats = glue.stats
    assert stats["kl_samples"] == 25
    assert stats["reward_samples"] == 25
    assert stats["pred_error_samples"] == 25
    print(f"  Glue stats: {stats}")

    print("  [PASS] Glue Layer")


def run_all_tests():
    """Run all component tests."""
    print("=" * 60)
    print("  Developmental AI — Component Tests")
    print("=" * 60)

    tests = [
        ("RSSM World Model", test_rssm_world_model),
        ("Replay Buffer", test_replay_buffer),
        ("ICM Curiosity", test_icm_curiosity),
        ("Knowledge Graph", test_knowledge_graph),
        ("Skill Bank", test_skill_bank),
        ("Policy", test_policy),
        ("Environment Wrapper", test_environment_wrapper),
        ("Graph Embeddings", test_graph_embeddings),
        ("Symbolic Decoder", test_symbolic_decoder),
        ("Knowledge Integrator", test_knowledge_integrator),
        ("Glue Layer", test_glue_layer),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTesting {name}...")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
