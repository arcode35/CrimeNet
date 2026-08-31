import Image from "next/image";
import crimeSenseLogo from "@/assets/crimesense_logo_exact_cropped.svg";

export function BrandLogo({
  className,
  priority = false,
}: {
  className?: string;
  priority?: boolean;
}) {
  return (
    <Image
      src={crimeSenseLogo}
      alt="CrimeSense"
      width={1397}
      height={358}
      className={className}
      priority={priority}
      unoptimized
    />
  );
}
