# Review System Guide

This folder contains **deep-dive questions and answers** from actual learning sessions, complementing the structured quizzes in YAML files.

## Purpose

While YAML quizzes test core understanding, review FAQs capture:
- Specific confusions you encountered
- Edge cases and nuances
- Connections between concepts
- Real-world applications you discovered

## Structure
```
reviews/
├── 01_math_foundations_faq.md
├── 02_deep_learning_faq.md
├── 03_transformer_faq.md
├── 04_llm_faq.md
├── 05_agent_faq.md
└── README.md (this file)
```

Each FAQ file follows this template:
```markdown
# [Domain Name] - Review Questions

## YYYY-MM-DD: [Study Session Topic]

### Q: [Your question]

**Context**: [What you were studying]
**Source**: [Link to conversation/paper/video]

**Answer**: 
- [Key point 1]
- [Key point 2]

**Related nodes**: 
- `domain:node_id`

**Confidence**: ⭐⭐⭐ (1-3 stars)

---
```

## Workflow

### During Study Session

1. When you encounter something unclear, **pause and document**:
   - What exactly is confusing?
   - What context led to this question?
   
2. After getting clarification (Claude, paper, mentor), **record the answer**:
   - Write in your own words
   - Include multiple angles if helpful
   - Link to the source for future reference

3. **Link to graph nodes**:
   - Which concepts in the YAML graph is this related to?
   - Use `domain:node_id` format

4. **Set confidence level**:
   - ⭐ (1/3): Just learned, still shaky
   - ⭐⭐ (2/3): Understood, can explain
   - ⭐⭐⭐ (3/3): Mastered, can teach others

### Weekly Review

Every week:
- Add new questions from study sessions
- Quick scan of last week's questions
- Update confidence if you've practiced

### Monthly Deep Review

Every month:
- Focus on ⭐ (low confidence) questions
- Re-test yourself without looking at answers
- Update confidence levels
- Archive or mark as mastered

## Examples

### Good Question Entry
```markdown
### Q: Why divide by sqrt(d_k) in scaled dot-product attention?

**Context**: Implementing attention from scratch, noticed variance issues with large d_k
**Source**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) Section 3.2.1

**Answer**: 
- Dot products grow in magnitude with dimensionality
- Large values push softmax into saturation (gradients vanish)
- sqrt(d_k) scaling keeps variance stable (~1) regardless of d_k

**Related nodes**: 
- `transformer:scaled_attention` - The core mechanism
- `math_foundations:variance` - Why variance matters
- `deep_learning:gradient_flow` - Gradient vanishing problem

**Confidence**: ⭐⭐⭐ (3/3)

**Review history**:
- 2026-01-01: First learned from paper
- 2026-01-15: Reviewed, implemented from scratch
- 2026-02-01: Confidence upgraded ⭐⭐ → ⭐⭐⭐
```

### Less Useful Entry (Avoid This)
```markdown
### Q: What is attention?

**Answer**: It's a mechanism to focus on important parts.

**Related nodes**: `transformer:attention`
```

**Why bad?**: Too vague, no context, no depth, just repeats definition.

## Tips

### Writing Good Questions

✅ **Good**: "Why does multi-head attention use multiple heads instead of one large attention?"
❌ **Bad**: "What is multi-head attention?"

✅ **Good**: "How does temperature scaling in softmax affect attention distribution sharpness?"
❌ **Bad**: "What is temperature?"

### Writing Good Answers

- **Use your own words**: Don't copy-paste definitions
- **Include intuition**: Why does this make sense?
- **Add examples**: When possible, use concrete scenarios
- **Link concepts**: How does this connect to other things you know?

### Managing Confidence

- **Be honest**: Low confidence is okay, it guides your review
- **Update regularly**: As you practice, confidence should increase
- **Don't inflate**: ⭐⭐⭐ means you can teach it confidently

### Source Attribution

Always include sources:
- Claude conversations: `[Claude chat](https://claude.ai/share/xxx)`
- Papers: `[Paper Title](arXiv link) Section X.Y`
- Videos: `[Video Title](YouTube link) @ timestamp`
- Books: `[Book Title] Chapter X, Page Y`

## Integration with YAML Graphs

Review FAQs **complement** YAML quizzes:

| YAML Quizzes | Review FAQs |
|--------------|-------------|
| Breadth coverage | Depth exploration |
| Standard questions | Personal questions |
| Required for all | Only where you went deep |
| Static | Evolving |

**Example**:
- YAML quiz: "What is the purpose of positional encoding?"
- Review FAQ: "Why do transformers use sinusoidal positional encoding instead of learned embeddings? When does each work better?"

## Maintenance

### When to Archive

If a question reaches ⭐⭐⭐ and stays there for 3+ months:
- Move to `_archived/` subfolder (create if needed)
- Keep for historical reference
- Frees up active review list

### When to Delete

Generally, **never delete**:
- Questions document your learning journey
- Future you might find them insightful
- Low disk space cost

## Future Enhancements

- [ ] CLI tool: `python review.py add --domain transformer`
- [ ] Automated confidence decay (if not reviewed in X days)
- [ ] Export to Anki flashcards with spaced repetition
- [ ] Link checker (verify source URLs still work)
- [ ] Statistics: questions per domain, confidence distribution

---

**Remember**: The best review system is one you actually use. Start simple, stay consistent.