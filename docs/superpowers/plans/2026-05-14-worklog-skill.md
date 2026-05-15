# Worklog Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Claude skill that automatically detects and summarizes daily work activities from user conversations, generating structured daily and weekly Markdown reports.

**Architecture:** The skill operates entirely through SKILL.md instructions. It instructs Claude to monitor conversations for work completion signals, append detected items to a daily draft file, and generate formatted Markdown reports on demand. No bundled scripts are needed — all file operations use Claude's built-in Read/Write/Bash tools.

**Tech Stack:** Claude skill format (SKILL.md + evals.json), Markdown, local filesystem.

---

## File Structure

```
/tmp/worklog-skill/
├── SKILL.md              # Core skill instructions (frontmatter + body)
└── evals/
    └── evals.json        # Test prompts for validation
```

No `scripts/`, `references/`, or `assets/` are needed for this skill. All logic is expressed in SKILL.md instructions.

---

## Task 1: Create Skill Directory and Skeleton

**Files:**
- Create: `/tmp/worklog-skill/SKILL.md`
- Create: `/tmp/worklog-skill/evals/evals.json`

- [ ] **Step 1: Create directory structure**

Run:
```bash
mkdir -p /tmp/worklog-skill/evals
```

- [ ] **Step 2: Write SKILL.md frontmatter**

Write to `/tmp/worklog-skill/SKILL.md`:
```markdown
---
name: worklog-skill
description: >
  自动记录用户与 Claude 的协作工作内容，生成日报和周报 Markdown 文件。
  当用户提到"日报"、"周报"、"今天做了什么"、"这周做了什么"、
  "记录"、"记一下"等指令时触发。
  当对话中出现工作完成信号（修复了 bug、实现了功能、提交了代码等）时，
  自动追加到当日工作草稿。
  务必使用此 skill 处理所有工作日志、日报、周报、工作总结相关的请求，
  即使用户没有明确说"日报"二字，只要涉及"今天做了什么""总结一下"等场景，
  都应调用此 skill。
---
```

- [ ] **Step 3: Write initial evals.json skeleton**

Write to `/tmp/worklog-skill/evals/evals.json`:
```json
{
  "skill_name": "worklog-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "",
      "expected_output": "",
      "files": []
    }
  ]
}
```

- [ ] **Step 4: Commit directory creation**

Run:
```bash
cd /tmp/worklog-skill && git init && git add . && git commit -m "chore: init worklog skill directory"
```

---

## Task 2: Write SKILL.md — Core Instructions (Part 1)

**Files:**
- Modify: `/tmp/worklog-skill/SKILL.md`

Append to SKILL.md:
```markdown
# Worklog Skill

## Overview

本 skill 用于自动追踪和记录用户在与 Claude 协作过程中的工作内容，
生成结构化的日报和周报 Markdown 文件。

所有工作记录保存在 `~/worklogs/` 目录下：
- 草稿：`.draft-YYYY-MM-DD.md`（当日所有会话共享，追加写入）
- 日报：`YYYY-MM-DD-daily.md`
- 周报：`YYYY-weekNN.md`

## Trigger Conditions

当用户说出以下任意意图时，触发本 skill：
- "生成日报"、"写日报"、"今天做了什么"
- "生成周报"、"这周总结"、"周报"
- "记录：xxx"、"记一下，xxx"
- "看看草稿"、"今天记录了什么"
- "把日报第三条删掉"、"修改日报"
- "清空今天记录"
- 以及任何涉及"工作总结""汇报""review""review 一下"等场景

## Auto-Detection Rules

在对话过程中，持续监听工作完成信号。

### 小事（静默记录）
以下类型的工作完成后，直接追加到当日草稿，不询问用户：
- 配置/参数调整
- 文档/注释更新
- 单文件内的小型重构
- 临时脚本或调试代码

### 大事（用户确认）
以下类型的工作完成后，先询问用户是否记录：
- Bug 修复
- 新功能完成
- 跨文件重构
- 代码合并/提交
- 性能优化

### 状态过滤
优先匹配**完成态**表述：
- "已经/刚刚/解决了/完成了/提交了/合并了/优化了"

忽略**将来态**和**假设态**：
- "准备/计划/明天/下次/如果/也许/可能/试试"

如果 5 分钟内同一话题被推翻，自动标记前一条为无效。

## Draft File Format

草稿文件为内部格式，追加写入：
```markdown
- [auto] 14:32 修复了登录超时问题（backend/auth.py）
- [manual] 15:10 记录了：下午要开会讨论方案
- [confirmed] 16:45 实现了动态帧采样逻辑
```

生成日报时，去掉标记，整理为用户友好的格式。
```

- [ ] **Step 1: Append Part 1 to SKILL.md**

Use Write or Edit to append the content above to `/tmp/worklog-skill/SKILL.md`.

- [ ] **Step 2: Verify SKILL.md exists and is well-formed**

Run:
```bash
head -20 /tmp/worklog-skill/SKILL.md
```
Expected: Frontmatter block with `name: worklog-skill`.

- [ ] **Step 3: Commit**

```bash
cd /tmp/worklog-skill && git add SKILL.md && git commit -m "feat: add core instructions part 1"
```

---

## Task 3: Write SKILL.md — Report Generation (Part 2)

**Files:**
- Modify: `/tmp/worklog-skill/SKILL.md`

Append to SKILL.md:
```markdown
## Daily Report Generation

当用户触发"生成日报"时，执行以下步骤：

1. **读取草稿**：读取 `~/worklogs/.draft-YYYY-MM-DD.md`（当前日期）。
2. **分析对话**：检查当前会话对话中是否有草稿未覆盖的工作内容，补全到列表。
3. **去重合并**：按时间排序，去除重复项（相同事项只保留一条）。
4. **自动分类**：基于关键词判断分类：
   - `[功能]`：实现了、新增了、添加了
   - `[Bug修复]`：修复了、解决了
   - `[优化]`：优化了、重构了、提升了、降低了
   - `[文档]`：更新了、完善了、编写了
   - `[其他]`：不属于以上类别
5. **填充模板**：按以下模板生成日报：

```markdown
# 工作日报 — YYYY-MM-DD

## 今日完成

### [功能]
- xxx

### [Bug修复]
- xxx

### [优化]
- xxx

### [文档]
- xxx

### [其他]
- xxx

## 进行中
- xxx（如草稿中有"正在进行"或"未完成"的描述）

## 明日计划
- xxx（如草稿中有"明天要做"的描述）

## 备注
- 自动记录 X 条，手动记录 Y 条
```

6. **输出文件**：将日报写入 `~/worklogs/YYYY-MM-DD-daily.md`。
7. **清空草稿**：删除或清空当日草稿文件（可选保留 `.bak` 备份）。
8. **报告路径**：仅向用户报告文件路径，**不要在对话中打印日报全文**。

## Weekly Report Generation

当用户触发"生成周报"时，执行以下步骤：

1. **扫描日报**：读取 `~/worklogs/` 下本周所有 `*-daily.md` 文件（根据文件名日期判断）。
2. **按分类汇总**：将各日报中的 `[功能]`、`[Bug修复]`、`[优化]` 等分类汇总。
3. **去重**：同一事项在不同日报中重复出现时只保留一条。
4. **统计**：计算本周各分类的数量。
5. **汇总计划**：汇总所有日报中的"明日计划"作为"下周计划"。
   - 若某条"明日计划"在下一天没有对应完成记录，标记为"未完成"。
6. **填充模板**：

```markdown
# 工作周报 — YYYY-MM-DD 至 YYYY-MM-DD

## 本周概览
- 完成功能：X 项
- 修复 Bug：Y 个
- 性能优化：Z 项
- 文档/其他：W 项

## 详细内容

### 功能开发
- MM-DD：xxx

### Bug修复
- MM-DD：xxx

### 优化
- MM-DD：xxx

## 下周计划
- xxx
```

7. **输出文件**：将周报写入 `~/worklogs/YYYY-weekNN.md`。
8. **报告路径**：仅向用户报告文件路径。

## Edge Cases

- **草稿不存在**：首次使用时自动创建空草稿。
- **同一天多次生成日报**：覆盖已有日报文件。
- **跨午夜**：以系统时间为准，23:59 和 00:01 分属两天。
- **用户否认**：支持"这不是工作""忽略刚才那条"撤销自动检测。
- **非工作会话**：无工作完成信号的内容不会触发记录。
```

- [ ] **Step 1: Append Part 2 to SKILL.md**

- [ ] **Step 2: Verify SKILL.md is under 500 lines**

Run:
```bash
wc -l /tmp/worklog-skill/SKILL.md
```
Expected: Under 500 lines. If over, consider splitting into reference files.

- [ ] **Step 3: Commit**

```bash
cd /tmp/worklog-skill && git add SKILL.md && git commit -m "feat: add report generation instructions"
```

---

## Task 4: Write evals.json

**Files:**
- Modify: `/tmp/worklog-skill/evals/evals.json`

- [ ] **Step 1: Write test prompts**

Rewrite `/tmp/worklog-skill/evals/evals.json` with the following content:
```json
{
  "skill_name": "worklog-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "记录：下午三点要开会讨论VLM prompt优化方案",
      "expected_output": "Claude 应将该事项追加到当日草稿文件 ~/worklogs/.draft-YYYY-MM-DD.md，并确认已记录。",
      "files": ["~/worklogs/.draft-YYYY-MM-DD.md"]
    },
    {
      "id": 2,
      "prompt": "刚刚修复了backend/config.py里的帧采样不均匀问题，把max_frames逻辑改成了基于window_delay动态计算。",
      "expected_output": "Claude 应检测到 Bug 修复信号，询问用户是否记录，确认后将事项追加到草稿。",
      "files": ["~/worklogs/.draft-YYYY-MM-DD.md"]
    },
    {
      "id": 3,
      "prompt": "生成今天的日报",
      "expected_output": "Claude 应读取当日草稿，分析当前会话，去重分类后生成格式化日报，保存到 ~/worklogs/YYYY-MM-DD-daily.md，并仅报告文件路径。",
      "files": ["~/worklogs/YYYY-MM-DD-daily.md"]
    },
    {
      "id": 4,
      "prompt": "这周做了什么",
      "expected_output": "Claude 应扫描本周所有日报，按分类汇总生成周报，保存到 ~/worklogs/YYYY-weekNN.md，并仅报告文件路径。",
      "files": ["~/worklogs/YYYY-weekNN.md"]
    },
    {
      "id": 5,
      "prompt": "把日报里的'重构了camera_manager'那条删掉",
      "expected_output": "Claude 应读取当天日报文件，删除指定条目，保存修改后的文件。",
      "files": ["~/worklogs/YYYY-MM-DD-daily.md"]
    }
  ]
}
```

- [ ] **Step 2: Validate JSON syntax**

Run:
```bash
python -m json.tool /tmp/worklog-skill/evals/evals.json > /dev/null && echo "JSON valid"
```
Expected: `JSON valid`

- [ ] **Step 3: Commit**

```bash
cd /tmp/worklog-skill && git add evals/evals.json && git commit -m "test: add eval prompts"
```

---

## Task 5: Run Evals (Skill-Creator Testing Loop)

**Files:**
- Create: `/tmp/worklog-skill-workspace/` (results directory)
- Read: `/tmp/worklog-skill/evals/evals.json`
- Read: `/tmp/worklog-skill/SKILL.md`

- [ ] **Step 1: Create workspace directory**

Run:
```bash
mkdir -p /tmp/worklog-skill-workspace/iteration-1
```

- [ ] **Step 2: For each eval, spawn WITH-SKILL and WITHOUT-SKILL subagents**

For eval N in evals.json:
1. Create directory `/tmp/worklog-skill-workspace/iteration-1/eval-N/`
2. Spawn subagent A (WITH skill):
   - Load skill from `/tmp/worklog-skill/SKILL.md`
   - Execute prompt from evals.json
   - Save outputs to `/tmp/worklog-skill-workspace/iteration-1/eval-N/with_skill/outputs/`
3. Spawn subagent B (WITHOUT skill / baseline):
   - Execute same prompt without loading skill
   - Save outputs to `/tmp/worklog-skill-workspace/iteration-1/eval-N/without_skill/outputs/`
4. Write `eval_metadata.json` for each eval:
   ```json
   {
     "eval_id": N,
     "eval_name": "...",
     "prompt": "...",
     "assertions": []
   }
   ```

- [ ] **Step 3: While runs are in progress, draft assertions**

For each eval, define 2-3 quantitative assertions. Examples:
- "draft_file_created": Checks if `~/worklogs/.draft-*.md` was created.
- "daily_report_format": Checks if daily report contains sections "今日完成", "进行中".
- "no_full_content_in_chat": Checks if response only contains file path, not full report.
- "weekly_has_summary": Checks if weekly report contains "本周概览" with counts.

Update `eval_metadata.json` and `evals/evals.json` with assertions.

- [ ] **Step 4: As runs complete, capture timing data**

For each completed subagent task, save `timing.json`:
```json
{
  "total_tokens": 12345,
  "duration_ms": 45678,
  "total_duration_seconds": 45.7
}
```

- [ ] **Step 5: Grade each run**

Spawn grader subagents (or grade inline) that evaluate each assertion against outputs.
Save `grading.json` with exact fields: `text`, `passed`, `evidence`.

- [ ] **Step 6: Aggregate benchmark**

Run:
```bash
python -m scripts.aggregate_benchmark /tmp/worklog-skill-workspace/iteration-1 --skill-name worklog-skill
```
Expected output: `benchmark.json` and `benchmark.md` with pass_rate, time, tokens.

- [ ] **Step 7: Launch eval viewer**

Run:
```bash
nohup python /path/to/skill-creator/eval-viewer/generate_review.py \
  /tmp/worklog-skill-workspace/iteration-1 \
  --skill-name "worklog-skill" \
  --benchmark /tmp/worklog-skill-workspace/iteration-1/benchmark.json \
  > /dev/null 2>&1 &
echo $! > /tmp/worklog-viewer.pid
```

- [ ] **Step 8: User review**

Tell user: "Eval viewer is running. Review the outputs and benchmark, then come back with feedback."

---

## Task 6: Iterate and Improve

**Files:**
- Modify: `/tmp/worklog-skill/SKILL.md`
- Modify: `/tmp/worklog-skill/evals/evals.json`
- Create: `/tmp/worklog-skill-workspace/iteration-2/` (and onward)

- [ ] **Step 1: Read user feedback**

Read `/tmp/worklog-skill-workspace/iteration-1/feedback.json` (downloaded from viewer).

- [ ] **Step 2: Generalize from feedback**

Apply improvements to SKILL.md based on user feedback. Rules:
- Fix specific failures observed in test runs.
- Generalize fixes so they apply to future invocations, not just the test cases.
- Remove instructions that caused unproductive behavior (read transcripts, not just outputs).
- Explain the "why" behind each instruction.

- [ ] **Step 3: Rerun evals in iteration-2**

Repeat Task 5 in `/tmp/worklog-skill-workspace/iteration-2/`.
Baseline remains `without_skill`.

- [ ] **Step 4: Launch viewer with previous comparison**

Run:
```bash
nohup python /path/to/skill-creator/eval-viewer/generate_review.py \
  /tmp/worklog-skill-workspace/iteration-2 \
  --skill-name "worklog-skill" \
  --benchmark /tmp/worklog-skill-workspace/iteration-2/benchmark.json \
  --previous-workspace /tmp/worklog-skill-workspace/iteration-1 \
  > /dev/null 2>&1 &
```

- [ ] **Step 5: Repeat until satisfied**

Continue iterating (iteration-3, etc.) until:
- User says they're happy.
- All feedback is empty.
- No meaningful progress is being made.

---

## Task 7: Description Optimization

**Files:**
- Modify: `/tmp/worklog-skill/SKILL.md` (frontmatter only)

- [ ] **Step 1: Generate trigger eval queries**

Create 20 realistic queries (10 should-trigger, 10 should-not-trigger).
Save to `/tmp/worklog-skill-workspace/trigger_evals.json`.

- [ ] **Step 2: Review with user**

Use skill-creator's HTML template (`assets/eval_review.html`) to let user review and edit queries.

- [ ] **Step 3: Run optimization loop**

```bash
python -m scripts.run_loop \
  --eval-set /tmp/worklog-skill-workspace/trigger_evals.json \
  --skill-path /tmp/worklog-skill \
  --model claude-opus-4-7 \
  --max-iterations 5 \
  --verbose
```

- [ ] **Step 4: Apply best description**

Take `best_description` from JSON output and update SKILL.md frontmatter.

---

## Task 8: Package and Present

**Files:**
- Create: `/tmp/worklog-skill.skill` (packaged file)

- [ ] **Step 1: Package skill**

```bash
python -m scripts.package_skill /tmp/worklog-skill
```

- [ ] **Step 2: Report path to user**

Tell user: "Skill packaged at `/tmp/worklog-skill.skill`. You can install it with Claude Code or Claude.ai."

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] 自动检测规则（大事/小事/状态过滤）
- [x] 草稿文件操作（读取/追加/清空）
- [x] 日报生成（读取草稿 + 分析对话 + 去重分类 + 模板填充）
- [x] 周报生成（扫描日报 + 汇总 + 统计 + 计划追踪）
- [x] 边界情况（草稿不存在、跨午夜、用户否认等）
- [x] 输出方式（直接保存文件，不打印全文）
- [x] 触发指令（日报、周报、记录、查看草稿、修改）

**2. Placeholder scan:**
- [x] 无 TBD/TODO
- [x] 无 "add appropriate error handling" 等模糊表述
- [x] 所有步骤包含具体命令或代码

**3. Type consistency:**
- [x] 文件命名格式一致（`.draft-YYYY-MM-DD.md`、`YYYY-MM-DD-daily.md`、`YYYY-weekNN.md`）
- [x] 分类标签一致（`[功能]`、`[Bug修复]`、`[优化]`、`[文档]`、`[其他]`）
