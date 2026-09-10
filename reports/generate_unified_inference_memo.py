"""Build the unified-inference evaluation memo as a PDF."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

OUT = '/home/ubuntu/onet/reports/unified_inference_evaluation.pdf'

INK = colors.HexColor('#1a1a1a')
MUTED = colors.HexColor('#5f6b76')
RULE = colors.HexColor('#c8ced6')
BAND = colors.HexColor('#eef1f5')
ACCENT = colors.HexColor('#1f4e79')
GOOD = colors.HexColor('#1e6b3a')
BAD = colors.HexColor('#98332b')

ss = getSampleStyleSheet()


def S(name, **kw):
    kw.setdefault('parent', ss['Normal'])
    return ParagraphStyle(name, **kw)


body = S('body', fontName='Helvetica', fontSize=9.6, leading=14.2, textColor=INK,
         alignment=TA_JUSTIFY, spaceAfter=7)
lead = S('lead', parent=body, fontSize=10.2, leading=15.2, spaceAfter=9)
h1 = S('h1', fontName='Helvetica-Bold', fontSize=12.2, leading=15, textColor=ACCENT,
       spaceBefore=15, spaceAfter=6)
h2 = S('h2', fontName='Helvetica-Bold', fontSize=10, leading=13.5, textColor=INK,
       spaceBefore=9, spaceAfter=3)
small = S('small', fontName='Helvetica', fontSize=8.3, leading=11.6, textColor=MUTED,
          spaceAfter=5)
cell = S('cell', fontName='Helvetica', fontSize=8.4, leading=11)
cellb = S('cellb', fontName='Helvetica-Bold', fontSize=8.4, leading=11)
cellh = S('cellh', fontName='Helvetica-Bold', fontSize=8.2, leading=10.6,
          textColor=colors.white)
mono = S('mono', fontName='Courier', fontSize=8.2, leading=11)
bullet = S('bullet', parent=body, leftIndent=13, bulletIndent=3, spaceAfter=4.5,
           alignment=0)


def P(t, st=body):
    return Paragraph(t, st)


def B(t):
    return Paragraph(t, bullet, bulletText='•')


def rule(w=0.6, c=RULE, sb=2, sa=8):
    return HRFlowable(width='100%', thickness=w, color=c, spaceBefore=sb, spaceAfter=sa)


def table(rows, widths, aligns=None, shade=()):
    """shade = row indices (1-based over rows incl. header) drawn banded and bold."""
    data = [[Paragraph(c, cellh) for c in rows[0]]]
    for i, r in enumerate(rows[1:], start=1):
        st = cellb if i in shade else cell
        data.append([Paragraph(c, st) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, -2), 0.4, RULE),
        ('LINEBELOW', (0, -1), (-1, -1), 0.7, RULE),
    ]
    for i in shade:
        style.append(('BACKGROUND', (0, i), (-1, i), BAND))
    if aligns:
        for col, a in aligns.items():
            style.append(('ALIGN', (col, 1), (col, -1), a))
    t.setStyle(TableStyle(style))
    return t


def header(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.9 * inch, LETTER[1] - 0.58 * inch,
                      'Fish Species Recognition Challenge  |  Internal evaluation memo')
    canvas.drawRightString(LETTER[0] - 0.9 * inch, LETTER[1] - 0.58 * inch, '8 August 2026')
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(0.9 * inch, LETTER[1] - 0.68 * inch, LETTER[0] - 0.9 * inch, LETTER[1] - 0.68 * inch)
    canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch, str(doc.page))
    canvas.restoreState()


story = []

story.append(Paragraph('Unified single-head inference: evaluation and recommendation',
                       S('title', fontName='Helvetica-Bold', fontSize=17, leading=20.5,
                         textColor=INK, spaceAfter=4)))
story.append(Paragraph(
    'Removing the separate seen/unseen classifier heads in favour of a two-stage '
    'shortlist-then-decide pipeline. Questions answered, changes made, measured results.',
    S('sub', fontName='Helvetica', fontSize=10, leading=14, textColor=MUTED, spaceAfter=8)))
story.append(rule(1.1, ACCENT, 0, 12))

# ---------------------------------------------------------------- summary
story.append(P(
    'The proposal was to drop the two classifier heads and run every evaluation image through one '
    'pipeline: stage 1 compares the image against vision prototypes and keeps the top-k most similar '
    'seen classes; stage 2 makes the final call by comparing the image against text prompts for those '
    'k classes plus all unseen classes. I built it, measured it four ways, and the answer is that it '
    'costs us accuracy rather than gaining any. Details below.', lead))

box = Table([[Paragraph(
    '<b>Bottom line.</b> The unified pipeline scores best on our internal holdout (57.02 vs 50.71) and '
    'worst on the eval set, projecting <b>49.35% at best against the 51.44% we have already banked</b>. '
    'The holdout is actively misleading on this axis: it ranks our two live configurations in the wrong '
    'order by five points. No submission slot was spent.', body)]],
    colWidths=[6.7 * inch])
box.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), BAND),
    ('LINEBEFORE', (0, 0), (0, -1), 2.5, ACCENT),
    ('LEFTPADDING', (0, 0), (-1, -1), 11), ('RIGHTPADDING', (0, 0), (-1, -1), 11),
    ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(box)

# ---------------------------------------------------------------- Q&A
story.append(P('Questions answered', h1))

story.append(P('1. Are the prototypes pre-extracted from all training images?', h2))
story.append(P(
    'Yes. Each class prototype is the mean of the embeddings of every training image belonging to that '
    'class, computed once from cached embedding files rather than at inference time. Two things worth '
    'knowing that the question implies but the code does not: the prototype pool is <b>64,259</b> images, '
    'not the full 99,979, because we intersect across the seven encoder embedding files and only images '
    'present in all of them are usable; and the seen-class score is not the prototype alone. It is a sum '
    'of three terms:'))
story.append(Paragraph('score = prototype_cosine  +  2.0 x cmax  +  4.0 x taxon_text_cosine', mono))
story.append(Spacer(1, 5))
story.append(P(
    'where <b>cmax</b> is the highest cosine against any single training image of that class. That '
    'distinction matters for the second question.'))

story.append(P('2. Should cosine scores be re-weighted by training count instead of using the '
               'seen/unseen split explicitly?', h2))
story.append(P(
    'The instinct is right and the effect is larger than one would guess. Training counts are extremely '
    'skewed: minimum 2, <b>median 2</b>, maximum 281. Half the seen classes have exactly two images, so '
    'this is not a rounding concern.'))
story.append(P(
    'The cleanest place to apply it is the <b>cmax</b> term specifically, rather than the whole seen '
    'block. A maximum over n samples grows with n by construction, so a 281-image class gets 281 draws '
    'at the maximum while a 2-image class gets two. That is an upward bias for head classes independent '
    'of any true similarity, and it is exactly the kind of unfairness the re-weighting is meant to '
    'correct. The mean-prototype term is a different story, since its reliability depends on within-class '
    'spread and the sign is not obvious in advance.'))
story.append(P(
    'A fixed top-2 mean per class neutralises the max-of-n bias directly and is feasible for every class '
    'given the minimum count of 2. A helper for this already exists in the shared research utilities. '
    'This was deferred as requested and is not part of the results below.'))

story.append(P('3. On the rules', h2))
story.append(P(
    'The change was framed as fitting new rules. I re-fetched the live competition terms on 8 August and '
    'compared them against our cached copy from 23 July: the relevant clause is <b>unchanged, word for '
    'word</b>. There is no new rule here.'))
story.append(P(
    'More importantly, the current pipeline does not use split information. The clause prohibits using '
    'the provided split files to identify whether an evaluation image is seen or unseen, or to restrict '
    'the candidate classes. Our gate is computed from the image and text embeddings alone, and the '
    'seen/unseen class partition is derived from the training labels file, never from the split files. '
    'The rules explicitly permit "a novelty/gate signal computed from the image/text itself".'))
story.append(P(
    'So a unified pipeline is an architecture preference, not a compliance fix. That distinction is worth '
    'settling before we weigh the cost, because if it were a compliance fix we would have to accept the '
    'cost regardless.'))

# ---------------------------------------------------------------- changes
story.append(P('What was built', h1))
story.append(P(
    'One evaluation harness comparing four scoring architectures on identical inputs, identical class '
    'prototypes and identical text stacks, so the only variable is the pipeline shape. Stage 1 is the '
    'same vision-prototype comparison in every case; the arms differ only in what stage 2 scores with.'))

story.append(table([
    ['Arm', 'Stage-2 score function', 'Heads'],
    ['<b>A</b>', 'Current production. A gate routes each image to a seen head (prototype argmax) '
                 'or an unseen head (text plus external image prototypes).', 'Two'],
    ['<b>B</b>', 'The proposal as specified: image-to-text similarity only, over the top-k seen '
                 'classes plus all unseen classes.', 'One'],
    ['<b>C</b>', 'B plus the external image-prototype legs extended across the full class space, '
                 'so seen and unseen candidates receive the same evidence types.', 'One'],
    ['<b>D</b>', 'C plus one uniform reference-photo leg: same encoder and same statistic on both '
                 'sides, using a class\'s training images where it has them and external photos '
                 'where it does not.', 'One'],
], [0.45 * inch, 5.05 * inch, 0.62 * inch]))
story.append(Spacer(1, 7))
story.append(P(
    'Arm D deserves a note: it is not in the original proposal. Arms B and C both leave one side of the '
    'candidate pool with evidence the other side cannot have, which is the same asymmetry the proposal '
    'set out to remove. D is the version that genuinely applies one rule to every class, and it is the '
    'strongest unified design the current artifacts support.'))
story.append(P(
    'Two measurement surfaces were used. The internal holdout, where the rarest 20% of seen classes are '
    'withheld and treated as novel, gives labelled accuracy. The real 35,665-image evaluation batch has '
    'no labels, so it gives the routing diagnostic instead: what fraction of each folder\'s images land '
    'on a seen versus an unseen class. Folder membership is used for counting only and never enters any '
    'prediction.'))
story.append(P(
    'Image and flipped image was already in place before this work. Horizontal-flip test-time '
    'augmentation is the default in every extraction path, alongside an aspect-corrected second view '
    'added to match the elongated framing of the evaluation images.', small))

# ---------------------------------------------------------------- results
story.append(KeepTogether([P('Results', h1), P('Holdout accuracy, weighted overall', h2), table([
    ['Arm', 'Seen', 'Unseen', 'Overall'],
    ['A, production gate at f = 0.60', '69.37', '26.62', '50.71'],
    ['A, production gate at f = 0.72', '81.04', '23.21', '55.80'],
    ['<b>B, proposal as specified (k = 5)</b>', '78.77', '28.95', '<b>57.02</b>'],
    ['C, full-space image legs (k = 5)', '66.87', '41.54', '55.82'],
    ['D, uniform reference-photo leg (k = 5)', '67.01', '43.74', '56.86'],
], [3.1 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
    aligns={1: 'CENTER', 2: 'CENTER', 3: 'CENTER'}, shade=(3,))]))
story.append(Spacer(1, 6))
story.append(P(
    'On this evidence the proposal wins by 1.2 points and we should ship it. That reading is wrong, and '
    'the next table is why.'))

story.append(P('The holdout does not survive its own calibration check', h2))
story.append(P(
    'We have real scores for two configurations of the current pipeline, which lets us ask what the '
    'holdout says about a comparison whose answer we already know.'))
story.append(KeepTogether(table([
    ['Configuration', 'Holdout says', 'Reality says'],
    ['Production at f = 0.60', '50.71', '<b>51.44</b>'],
    ['Production at f = 0.72', '<b>55.80</b>', '50.84'],
    ['<b>Verdict</b>', '<b>f = 0.72 wins by 5.09</b>', '<b>f = 0.60 wins by 0.60</b>'],
], [2.7 * inch, 1.9 * inch, 1.9 * inch],
    aligns={1: 'CENTER', 2: 'CENTER'}, shade=(3,))))
story.append(Spacer(1, 6))
story.append(P(
    'The holdout orders the two configurations backwards, and by five points. The reason is structural: '
    'the holdout has no distribution shift, its stand-in novel classes are rare seen classes rather than '
    'genuinely unseen ones, and it therefore systematically rewards whichever pipeline commits harder to '
    'the seen route. Every unified arm does exactly that. A proxy that fails this badly on the one axis '
    'where we can check it cannot settle the question, and this is not the first time it has produced a '
    'result that reversed on submission.'))

story.append(P('Projection from the evaluation batch', h2))
story.append(P(
    'The usable signal is routing behaviour on the real images. The production configuration reproduces '
    'our banked submission <b>exactly, all 35,665 predictions identical</b>, which means the conditional '
    'accuracies calibrated from it are measured rather than assumed: 88.0% on seen images kept on the '
    'seen route, 26.0% on unseen images caught by the unseen route. Applying those to each arm\'s routing '
    'gives an upper bound, because it credits every arm with the production head\'s accuracy.'))
story.append(KeepTogether(table([
    ['Arm', 'Test imgs to<br/>seen class', 'Unseen imgs to<br/>unseen class', 'Projected<br/>overall',
     'vs banked<br/>51.44'],
    ['<b>A, production (banked)</b>', '86.7%', '74.5%', '<b>51.44</b>', 'anchor'],
    ['B, proposal as specified', '82.7%', '73.4%', '49.35', '<font color="#98332b">-2.10</font>'],
    ['C, full-space image legs', '73.4%', '84.4%', '45.97', '<font color="#98332b">-5.47</font>'],
    ['D, uniform reference-photo leg', '77.6%', '74.3%', '46.88', '<font color="#98332b">-4.56</font>'],
], [2.15 * inch, 1.15 * inch, 1.3 * inch, 1.0 * inch, 0.9 * inch],
    aligns={1: 'CENTER', 2: 'CENTER', 3: 'CENTER', 4: 'CENTER'}, shade=(1,))))
story.append(Spacer(1, 6))
story.append(P(
    'These are ceilings, and the method is known to be generous: applied to the f = 0.72 configuration it '
    'over-predicts the real score by 1.23 points. Even granting every unified arm that much slack, they '
    'land between 48 and 50.5, all below what we already have. The unified arms also decide seen classes '
    'without the prototype ensemble, so their true conditional accuracy is lower than the figure credited '
    'to them here.'))

# ---------------------------------------------------------------- why
story.append(P('Why it loses', h1))
story.append(P(
    'The two heads are not redundant machinery wrapped around one model. They carry different evidence, '
    'and a single score function cannot use both.'))
story.append(B(
    'The seen route runs a three-encoder prototype ensemble with a taxonomy-text anchor. Nothing '
    'equivalent exists on the unseen side, because unseen classes have no training images by definition.'))
story.append(B(
    'The unseen route leans on external photo prototypes. Those cover only 966 of 5,795 seen classes, '
    'so they cannot be applied evenly across the candidate pool as things stand.'))
story.append(B(
    'Any single score function therefore either drops the ensemble, which costs 4.0 points of seen '
    'routing in arm B, or lets the lopsided image evidence swamp the seen side, which costs 13.3 points '
    'in arm C.'))
story.append(Spacer(1, 3))
story.append(P(
    'Arm D closes the asymmetry as far as the current artifacts allow and matches production\'s unseen '
    'routing to within 0.2 points, 74.3 against 74.5. It still gives up <b>9.1 points</b> of seen routing. '
    'That gap is structural rather than a tuning shortfall: closing it requires re-embedding the external '
    'photo bank under the other two encoders so the unseen side gets the same ensemble. Those embeddings '
    'do not exist and represent new GPU work.'))
story.append(P(
    'The shortlist size is not a lever either. Moving k from 5 to the full seen space changes holdout '
    'overall by 0.28, from 57.02 to 56.74. Stage 1 is contributing almost nothing; the unified pipeline '
    'is close to a plain argmax over the whole class space.'))

# ---------------------------------------------------------------- traps
story.append(P('Two implementation traps', h1))
story.append(P(
    'Both of these are live in the earlier two-stage builder and are worth fixing there regardless of '
    'what we decide about the architecture, since anyone rebuilding it will hit them.'))
story.append(P(
    '<b>Masking absent classes to a large negative value deletes them.</b> Classes with no external '
    'prototype are masked to -1e4. Inside a dedicated unseen head that is harmless, since such a class '
    'was unreachable there anyway. In a unified score it removes the class from the entire candidate '
    'space, so a confident text match can never win. Absent evidence has to be neutral, not negative.'))
story.append(P(
    '<b>A normalised leg added to only one side of the partition decides the outcome by itself.</b> The '
    'normalisation contributes a large constant offset per row. If that offset lands on the unseen '
    'columns and not the seen ones, the comparison is settled before any similarity is considered. In '
    'testing this produced a configuration that sent 99.7% of test images to seen classes and used a '
    'total of 94 unseen classes out of 11,598. That is a bug signature, not a result about unified '
    'inference, and it should not be quoted as one.'))

# ---------------------------------------------------------------- recommendation
story.append(P('Recommendation', h1))
story.append(P(
    'Keep the current two-head pipeline. It is compliant as it stands, and the unified alternative costs '
    'somewhere between two and five points against a banked result. No submission slot was spent on this '
    'and none should be; the previously staged two-stage packages belong to the same family as arm B and '
    'should not be submitted either.'))
story.append(P('If we want to revisit it, two things would have to change first:'))
story.append(B(
    'The external photo bank re-embedded under the remaining two encoders, so the unseen side of a '
    'unified score carries the same ensemble the seen side does. This is the only path that closes '
    'arm D\'s 9.1-point routing gap.'))
story.append(B(
    'A better proxy. The current holdout cannot rank routing architectures, and until something replaces '
    'it every change of this class has to be argued from routing behaviour and conditional accuracy '
    'rather than from proxy accuracy.'))
story.append(Spacer(1, 4))
story.append(P(
    'The count re-weighting question is separate from all of this and remains open on its own merits. It '
    'applies to the seen-class scoring inside whichever architecture we keep, and the max-of-n bias in '
    'the cmax term is a real and correctable effect. Worth testing next.'))

story.append(rule(0.6, RULE, 12, 6))
story.append(P(
    'Harness and result files: research/eval_unified_2stage.py, with the plain invocation producing '
    'holdout accuracy and the --real flag producing the routing pass. Numbers land in '
    'outputs/unified_2stage_holdout.json and outputs/unified_2stage_real_routing.json. Findings recorded '
    'in the project handoff under 8 August.', small))

doc = BaseDocTemplate(OUT, pagesize=LETTER, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                      topMargin=0.85 * inch, bottomMargin=0.8 * inch,
                      title='Unified single-head inference: evaluation and recommendation',
                      author='')
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')
doc.addPageTemplates([PageTemplate(id='main', frames=[frame], onPage=header)])
doc.build(story)
print('wrote', OUT)
