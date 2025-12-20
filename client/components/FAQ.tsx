import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

export default function FAQSection() {
  return (
    <section className="relative mx-auto mb-24 max-w-6xl px-6">
      <h2 className="mb-4 text-2xl font-semibold text-white">Frequently Asked Questions</h2>
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="item-1">
            <AccordionTrigger>Which video formats are supported?</AccordionTrigger>
            <AccordionContent>MP4 is recommended for best compatibility.</AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-2">
            <AccordionTrigger>Does it work in real-time?</AccordionTrigger>
            <AccordionContent>Processing happens server-side frame-by-frame with streaming updates to the dashboard.</AccordionContent>
          </AccordionItem>
          <AccordionItem value="item-3">
            <AccordionTrigger>How accurate is number plate recognition?</AccordionTrigger>
            <AccordionContent>Accuracy depends on camera angle, plate visibility, and lighting; EasyOCR attempts multiple retries per vehicle.</AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </section>
  );
}
