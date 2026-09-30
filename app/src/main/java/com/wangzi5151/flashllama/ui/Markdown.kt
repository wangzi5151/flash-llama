package com.wangzi5151.flashllama.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ---------------------------------------------------------------------------
// A small, dependency-free Markdown renderer covering the constructs that LLMs
// emit most often: headings, bold/italic/strike, inline + fenced code, lists,
// quotes, horizontal rules and tables.
// ---------------------------------------------------------------------------

private sealed interface MdBlock {
    data class Heading(val level: Int, val text: String) : MdBlock
    data class Code(val code: String) : MdBlock
    data class Bullet(val text: String, val marker: String, val indent: Int) : MdBlock
    data class Quote(val text: String) : MdBlock
    data class Table(val header: List<String>, val rows: List<List<String>>) : MdBlock
    data object Hr : MdBlock
    data class Para(val text: String) : MdBlock
}

private val HEADING = Regex("^(#{1,6})\\s+(.*)$")
private val HR = Regex("^\\s*([-*_])\\1{2,}\\s*$")
private val QUOTE = Regex("^\\s*>\\s?(.*)$")
private val BULLET = Regex("^(\\s*)[-*+]\\s+(.*)$")
private val ORDERED = Regex("^(\\s*)(\\d+)[.)]\\s+(.*)$")
private val FENCE = Regex("^```(.*)$")
private val TABLE_SEP = Regex("^\\s*\\|?\\s*:?-{1,}:?\\s*(\\|\\s*:?-{1,}:?\\s*)*\\|?\\s*$")

private fun isTableSeparator(line: String): Boolean =
    line.contains('-') && TABLE_SEP.matches(line)

private fun splitRow(line: String): List<String> {
    var s = line.trim()
    if (s.startsWith("|")) s = s.substring(1)
    if (s.endsWith("|")) s = s.dropLast(1)
    return s.split("|").map { it.trim() }
}

private fun isBlockStart(line: String): Boolean =
    HEADING.matches(line) || HR.matches(line) || QUOTE.matches(line) ||
        BULLET.matches(line) || ORDERED.matches(line) || FENCE.matches(line)

private fun parseMarkdown(src: String): List<MdBlock> {
    val lines = src.replace("\r\n", "\n").replace('\r', '\n').split("\n")
    val out = ArrayList<MdBlock>()
    var i = 0
    while (i < lines.size) {
        val line = lines[i]

        val fence = FENCE.find(line)
        if (fence != null) {
            i++
            val sb = StringBuilder()
            while (i < lines.size && !lines[i].trimStart().startsWith("```")) {
                sb.append(lines[i]).append('\n')
                i++
            }
            if (i < lines.size) i++
            out.add(MdBlock.Code(sb.toString().trimEnd('\n')))
            continue
        }

        if (line.contains('|') && i + 1 < lines.size && isTableSeparator(lines[i + 1])) {
            val header = splitRow(line)
            val rows = ArrayList<List<String>>()
            i += 2
            while (i < lines.size && lines[i].contains('|') && lines[i].isNotBlank()) {
                rows.add(splitRow(lines[i]))
                i++
            }
            out.add(MdBlock.Table(header, rows))
            continue
        }

        val mh = HEADING.find(line)
        if (mh != null) {
            out.add(MdBlock.Heading(mh.groupValues[1].length, mh.groupValues[2].trim()))
            i++
            continue
        }
        if (HR.matches(line)) {
            out.add(MdBlock.Hr)
            i++
            continue
        }
        val mq = QUOTE.find(line)
        if (mq != null) {
            out.add(MdBlock.Quote(mq.groupValues[1]))
            i++
            continue
        }
        val mo = ORDERED.find(line)
        if (mo != null) {
            out.add(MdBlock.Bullet(mo.groupValues[3], mo.groupValues[2] + ".", mo.groupValues[1].length / 2))
            i++
            continue
        }
        val mb = BULLET.find(line)
        if (mb != null) {
            out.add(MdBlock.Bullet(mb.groupValues[2], "•", mb.groupValues[1].length / 2))
            i++
            continue
        }
        if (line.isBlank()) {
            i++
            continue
        }

        val sb = StringBuilder(line)
        i++
        while (i < lines.size && lines[i].isNotBlank() && !isBlockStart(lines[i]) &&
            !(lines[i].contains('|') && i + 1 < lines.size && isTableSeparator(lines[i + 1]))
        ) {
            sb.append('\n').append(lines[i])
            i++
        }
        out.add(MdBlock.Para(sb.toString()))
    }
    return out
}

private val INLINE = Regex(
    "(`[^`]+`)" +
        "|(\\*\\*[^*]+\\*\\*)" +
        "|(__[^_]+__)" +
        "|(\\*[^*\\n]+\\*)" +
        "|(_[^_\\n]+_)" +
        "|(~~[^~]+~~)" +
        "|(\\[[^\\]]+\\]\\([^)\\s]+\\))"
)

@Composable
private fun inline(text: String, base: SpanStyle = SpanStyle()): AnnotatedString {
    val codeBg = MaterialTheme.colorScheme.surfaceVariant
    val linkColor = MaterialTheme.colorScheme.primary
    return remember(text, base, codeBg, linkColor) {
        buildAnnotatedString {
            var idx = 0
            for (m in INLINE.findAll(text)) {
                if (m.range.first > idx) append(text.substring(idx, m.range.first))
                val token = m.value
                when {
                    token.startsWith("`") ->
                        withStyle(base + SpanStyle(fontFamily = FontFamily.Monospace, background = codeBg)) {
                            append(token.substring(1, token.length - 1))
                        }
                    token.startsWith("**") ->
                        withStyle(base + SpanStyle(fontWeight = FontWeight.Bold)) {
                            append(token.substring(2, token.length - 2))
                        }
                    token.startsWith("__") ->
                        withStyle(base + SpanStyle(fontWeight = FontWeight.Bold)) {
                            append(token.substring(2, token.length - 2))
                        }
                    token.startsWith("~~") ->
                        withStyle(base + SpanStyle(textDecoration = androidx.compose.ui.text.style.TextDecoration.LineThrough)) {
                            append(token.substring(2, token.length - 2))
                        }
                    token.startsWith("[") -> {
                        val close = token.indexOf(']')
                        withStyle(base + SpanStyle(color = linkColor)) {
                            append(token.substring(1, close))
                        }
                    }
                    token.startsWith("*") ->
                        withStyle(base + SpanStyle(fontStyle = FontStyle.Italic)) {
                            append(token.substring(1, token.length - 1))
                        }
                    token.startsWith("_") ->
                        withStyle(base + SpanStyle(fontStyle = FontStyle.Italic)) {
                            append(token.substring(1, token.length - 1))
                        }
                    else -> append(token)
                }
                idx = m.range.last + 1
            }
            if (idx < text.length) append(text.substring(idx))
        }
    }
}

@Composable
fun MarkdownText(markdown: String, modifier: Modifier = Modifier) {
    val blocks = remember(markdown) { parseMarkdown(markdown) }
    Column(modifier) {
        blocks.forEachIndexed { index, block ->
            if (index > 0) Spacer(Modifier.height(6.dp))
            Block(block)
        }
    }
}

@Composable
private fun Block(block: MdBlock) {
    when (block) {
        is MdBlock.Heading -> Text(
            text = inline(block.text, SpanStyle(fontWeight = FontWeight.Bold)),
            fontSize = when (block.level) {
                1 -> 22.sp
                2 -> 20.sp
                3 -> 18.sp
                else -> 16.sp
            },
            style = MaterialTheme.typography.bodyLarge,
        )

        is MdBlock.Para -> Text(
            text = inline(block.text),
            style = MaterialTheme.typography.bodyMedium,
        )

        is MdBlock.Code -> Text(
            text = AnnotatedString(block.code),
            fontFamily = FontFamily.Monospace,
            fontSize = 13.sp,
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(8.dp))
                .padding(10.dp),
        )

        is MdBlock.Quote -> Row(
            Modifier
                .fillMaxWidth()
                .height(IntrinsicSize.Min)
        ) {
            Box(
                Modifier
                    .width(3.dp)
                    .fillMaxHeight()
                    .background(MaterialTheme.colorScheme.primary, RoundedCornerShape(2.dp))
            )
            Spacer(Modifier.width(8.dp))
            Text(
                text = inline(block.text, SpanStyle(fontStyle = FontStyle.Italic)),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        is MdBlock.Bullet -> Row(
            Modifier
                .fillMaxWidth()
                .padding(start = (block.indent * 16).dp)
        ) {
            Text(
                text = block.marker,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.width(24.dp),
            )
            Text(
                text = inline(block.text),
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f),
            )
        }

        MdBlock.Hr -> HorizontalDivider(Modifier.padding(vertical = 4.dp))

        is MdBlock.Table -> TableBlock(block)
    }
}

@Composable
private fun TableBlock(table: MdBlock.Table) {
    val cols = maxOf(
        table.header.size,
        table.rows.maxOfOrNull { it.size } ?: 0,
    )
    if (cols == 0) return
    val widths = IntArray(cols) { c ->
        maxOf(
            table.header.getOrNull(c)?.length ?: 0,
            table.rows.maxOfOrNull { it.getOrNull(c)?.length ?: 0 } ?: 0,
        ).coerceAtLeast(3)
    }
    Column(
        Modifier
            .horizontalScroll(rememberScrollState())
            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(8.dp))
            .padding(6.dp)
    ) {
        Row(Modifier.fillMaxWidth()) {
            for (c in 0 until cols) {
                Text(
                    text = inline(table.header.getOrNull(c) ?: "", SpanStyle(fontWeight = FontWeight.Bold)),
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier
                        .width((widths[c] * 9 + 20).dp)
                        .padding(horizontal = 6.dp, vertical = 4.dp),
                )
            }
        }
        HorizontalDivider()
        table.rows.forEach { row ->
            Row(Modifier.fillMaxWidth()) {
                for (c in 0 until cols) {
                    Text(
                        text = inline(row.getOrNull(c) ?: ""),
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier
                            .width((widths[c] * 9 + 20).dp)
                            .padding(horizontal = 6.dp, vertical = 4.dp),
                    )
                }
            }
            HorizontalDivider()
        }
    }
}
