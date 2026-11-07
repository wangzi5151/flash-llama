package com.wangzi5151.flashllama

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class PureLogicTest {

    @Test
    fun formatBytes_handlesNonPositiveAndUnknownValues() {
        assertEquals("-", formatBytes(0))
        assertEquals("-", formatBytes(-1))
        assertEquals("-", formatBytes(Long.MIN_VALUE))
    }

    @Test
    fun formatBytes_usesBinaryUnitsAndTwoDecimals() {
        assertEquals("1 B", formatBytes(1))
        assertEquals("512 B", formatBytes(512))
        assertEquals("1023 B", formatBytes(1023))
        assertEquals("1.00 KB", formatBytes(1024))
        assertEquals("1.50 KB", formatBytes(1536))
        assertEquals("1.00 MB", formatBytes(1024L * 1024))
        assertEquals("1.00 GB", formatBytes(1024L * 1024 * 1024))
        assertEquals("1.00 TB", formatBytes(1024L * 1024 * 1024 * 1024))
        assertEquals("8.00 TB", formatBytes(8L * 1024 * 1024 * 1024 * 1024))
    }

    @Test
    fun formatParams_handlesBillionsMillionsAndSmallValues() {
        assertEquals("-", formatParams(0))
        assertEquals("-", formatParams(-5))
        assertEquals("999", formatParams(999))
        assertEquals("1.00M", formatParams(1_000_000))
        assertEquals("2.50B", formatParams(2_500_000_000L))
    }

    @Test
    fun parseModelInfo_readsAllFields() {
        val info = parseModelInfo(
            "llama.gguf",
            "Llama 3.2 1B|1234567890|760000000|131072|32000|4096|b4321",
        )

        assertEquals("llama.gguf", info.name)
        assertEquals("Llama 3.2 1B", info.desc)
        assertEquals(1_234_567_890L, info.nParams)
        assertEquals(760_000_000L, info.sizeBytes)
        assertEquals(131072, info.nCtxTrain)
        assertEquals(32000, info.nVocab)
        assertEquals(4096, info.nCtx)
        assertEquals("b4321", info.llamaVersion)
    }

    @Test
    fun parseModelInfo_fallsBackForMissingFields() {
        val info = parseModelInfo("x.gguf", "only-a-description")

        assertEquals("only-a-description", info.desc)
        assertEquals(0L, info.nParams)
        assertEquals(0L, info.sizeBytes)
        assertEquals(0, info.nCtxTrain)
        assertEquals(0, info.nVocab)
        assertEquals(0, info.nCtx)
        assertEquals("", info.llamaVersion)
    }

    @Test
    fun parseModelInfo_rejectsInvalidNumbersAndEmptyDescription() {
        val info = parseModelInfo("x.gguf", "|abc|def|ghi|jkl|mno|")

        assertEquals("unknown", info.desc)
        assertEquals(0L, info.nParams)
        assertEquals(0L, info.sizeBytes)
        assertEquals(0, info.nCtxTrain)
        assertEquals(0, info.nVocab)
        assertEquals(0, info.nCtx)
        assertEquals("", info.llamaVersion)
    }

    @Test
    fun composePrompt_appliesTemplateAndSkipsBlankMessages() {
        val history = listOf(
            ChatMessage("system", "You are helpful."),
            ChatMessage("user", "Hello"),
            ChatMessage("assistant", ""),
            ChatMessage("user", "   "),
        )
        var capturedRoles: Array<String>? = null
        var capturedContents: Array<String>? = null
        var capturedAddAssistant: Boolean? = null

        val result = composePrompt(history, useChatTemplate = true) { roles, contents, add ->
            capturedRoles = roles
            capturedContents = contents
            capturedAddAssistant = add
            "<templated>"
        }

        assertEquals("<templated>", result)
        assertEquals(listOf("system", "user"), capturedRoles!!.toList())
        assertEquals(listOf("You are helpful.", "Hello"), capturedContents!!.toList())
        assertEquals(true, capturedAddAssistant)
    }

    @Test
    fun composePrompt_fallsBackToLastUserMessageWhenTemplateReturnsEmpty() {
        val history = listOf(
            ChatMessage("user", "first"),
            ChatMessage("assistant", "answer"),
            ChatMessage("user", "second"),
        )

        assertEquals("second", composePrompt(history, useChatTemplate = true) { _, _, _ -> "" })
        assertEquals("second", composePrompt(history, useChatTemplate = false, applyTemplate = null))
        assertEquals("second", composePrompt(history, useChatTemplate = true, applyTemplate = null))
    }

    @Test
    fun composePrompt_returnsEmptyWithoutUserMessage() {
        val history = listOf(
            ChatMessage("system", "system prompt"),
            ChatMessage("assistant", "hi"),
        )

        assertEquals("", composePrompt(history, useChatTemplate = false, applyTemplate = null))
        assertEquals("", composePrompt(history, useChatTemplate = true) { _, _, _ -> "" })
    }

    @Test
    fun pickThreadCount_fallsBackToAvailableProcessorsWhenNoFrequencies() {
        assertEquals(2, pickThreadCount(emptyList(), availableProcessors = 1))
        assertEquals(4, pickThreadCount(emptyList(), availableProcessors = 8))
        assertEquals(4, pickThreadCount(emptyList(), availableProcessors = 64))
    }

    @Test
    fun pickThreadCount_countsOnlyBigCores() {
        assertEquals(2, pickThreadCount(listOf(1_800_000, 1_700_000, 900_000, 800_000), 8))
        assertEquals(8, pickThreadCount(List(10) { 2_000_000L }, 10))
        assertEquals(2, pickThreadCount(listOf(2_000_000L), 8))
    }

    @Test
    fun defaultThreadCount_readsBigCoreFrequenciesFromCpuRoot() {
        val root = Files.createTempDirectory("flashllama-cpu").toFile()
        try {
            writeCpuFreq(root, "cpu0", 1_800_000)
            writeCpuFreq(root, "cpu1", 1_750_000)
            writeCpuFreq(root, "cpu2", 800_000)
            // Non-cpu entries and missing cpufreq files must be ignored.
            File(root, "cpufreq").mkdirs()
            File(root, "cpu3").mkdirs()

            assertEquals(2, defaultThreadCount(root))
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun defaultThreadCount_capsLargeClustersAtEight() {
        val root = Files.createTempDirectory("flashllama-cpu").toFile()
        try {
            repeat(12) { writeCpuFreq(root, "cpu$it", 2_000_000) }
            assertEquals(8, defaultThreadCount(root))
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun defaultThreadCount_usesRuntimeFallbackForEmptyRoot() {
        val root = Files.createTempDirectory("flashllama-cpu").toFile()
        try {
            val threads = defaultThreadCount(root)
            assertTrue("expected 2..4, got $threads", threads in 2..4)
        } finally {
            root.deleteRecursively()
        }
    }

    private fun writeCpuFreq(root: File, cpu: String, freq: Long) {
        val dir = File(root, "$cpu/cpufreq")
        assertTrue(dir.mkdirs())
        File(dir, "cpuinfo_max_freq").writeText("$freq\n")
    }
}
