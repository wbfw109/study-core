/*
 * This Kotlin source file was generated by the Gradle 'init' task.
 */
package wbfw109.app

import wbfw109.utilities.StringUtils

import org.apache.commons.text.WordUtils

/**
 * A group of *members*.
 *
 * This class has no useful logic; it's just a documentation example.
 *
 * @param T the type of a member in this group.
 * @property name the name of this group.
 * @constructor Creates an empty group.
 */
class Group<T>(val name: String) {
    /**
     * Adds a [member] to this group.
     * @return the new size of the group.
     */
    fun add(member: T): Int { return 3 }
}
fun main() {
    val tokens = StringUtils.split(MessageUtils.getMessage())
    val x: String = "abcde"
    println("$x")
    val result = StringUtils.join(tokens)
    println(WordUtils.capitalize(result))
    val group = Group<String>()
}


