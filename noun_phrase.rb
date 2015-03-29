=begin
  This is a demo of Noun Phrase Extraction algorithm. Using part of speech tags
  it'll extract the most likely noun phrases from a sentence.
=end

StanfordCoreNLP.jar_path = "./stanford-core-nlp/"
StanfordCoreNLP.model_path = "./stanford-core-nlp/"

PIPELINE = StanfordCoreNLP.load(:tokenize, :ssplit, :pos, :lemma, :parse, :ner, :dcoref)

class BaseExtractor
  def initialize
  end

  def extract(token_tags)
    entities = []
    current_entity = []
    token_tags.each do |tag|
      if yield(tag)
        current_entity << tag
      else
        unless current_entity.empty?
          entities << current_entity
          current_entity = []
        end
      end
    end

    entities << current_entity unless current_entity.empty?

    entities.map {|tags| tags.map {|t| t.value}.join(' ')}
  end
end

class TimeExtractor < BaseExtractor
  def extract(token_tags)
    super(token_tags) do |tag|
      tag.ne_tag == "TIME"
    end
  end
end

class NounPhraseExtractor < BaseExtractor
  def extract(token_tags)
    super(token_tags) do |tag|
      %w(DT JJ NN NNP CC NNS).include?(tag.pos)
    end
  end
end

class TweetParser
  TokenTag = Struct.new(:value, :pos, :ne_tag, :iob_tag)

  def self.parse(tweet)
    sentences = token_tuples(tweet)

    sentences.map do |sentence_tags|
      {
        nouns: NounPhraseExtractor.new.extract(sentence_tags),
        times: TimeExtractor.new.extract(sentence_tags)
      }
    end
  end

private
  def self.token_tuples(tweet)
    model = StanfordCoreNLP::Annotation.new(tweet)

    PIPELINE.annotate(model)
    tuples = []
    model.get(:sentences).each do |sentence|
      sentence_tuples = []
      sentence.get(:tokens).each do |token|
        sentence_tuples << TokenTag.new(token.get(:value).to_s, token.get(:part_of_speech).to_s, token.get(:named_entity_tag).to_s)
      end
      tuples << sentence_tuples
    end

    tuples
  end
end
